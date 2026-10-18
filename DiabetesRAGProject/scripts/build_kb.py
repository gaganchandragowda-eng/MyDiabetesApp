#!/usr/bin/env python3
"""
scripts/diabetes_kb_scraper.py

Professional web scraper for building diabetes management knowledge base from genuine medical sources.
Sources: CDC, WHO, Mayo Clinic, ADA, NIH/NIDDK, NHS, WebMD, Cleveland Clinic, Johns Hopkins

Features:
- Scrapes ONLY from verified authoritative medical sources
- Topic-based organization (diet, symptoms, treatment, complications, lifestyle)
- Content validation and medical relevance scoring
- Automatic link discovery within trusted domains
- Resume capability and progress tracking
- Structured JSON output for RAG ingestion
- Respects robots.txt and rate limiting
"""

import os
import re
import json
import time
import random
import hashlib
import logging
import argparse
from typing import List, Set, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin, parse_qs
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib import robotparser
from tqdm import tqdm

# -------------------------
# Genuine Medical Sources Configuration
# -------------------------
AUTHORITATIVE_SOURCES = {
    "cdc.gov": {
        "name": "Centers for Disease Control and Prevention",
        "trust_score": 1.0,
        "seeds": [
            "https://www.cdc.gov/diabetes/basics/index.html",
            "https://www.cdc.gov/diabetes/managing/index.html",
            "https://www.cdc.gov/diabetes/prevention/index.html",
            "https://www.cdc.gov/diabetes/library/features/role-of-insulin.html",
            "https://www.cdc.gov/diabetes/basics/symptoms.html"
        ]
    },
    "who.int": {
        "name": "World Health Organization",
        "trust_score": 1.0,
        "seeds": [
            "https://www.who.int/news-room/fact-sheets/detail/diabetes",
            "https://www.who.int/health-topics/diabetes"
        ]
    },
    "mayoclinic.org": {
        "name": "Mayo Clinic",
        "trust_score": 1.0,
        "seeds": [
            "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
            "https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451",
            "https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639",
            "https://www.mayoclinic.org/diseases-conditions/type-1-diabetes/symptoms-causes/syc-20353011"
        ]
    },
    "niddk.nih.gov": {
        "name": "National Institute of Diabetes and Digestive and Kidney Diseases",
        "trust_score": 1.0,
        "seeds": [
            "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes",
            "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes",
            "https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity",
            "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems",
            "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes"
        ]
    },
    "diabetes.org": {
        "name": "American Diabetes Association",
        "trust_score": 1.0,
        "seeds": [
            "https://diabetes.org/about-diabetes/type-1",
            "https://diabetes.org/about-diabetes/type-2",
            "https://diabetes.org/health-wellness/food-nutrition",
            "https://diabetes.org/health-wellness/fitness",
            "https://diabetes.org/health-wellness/medication-treatments"
        ]
    },
    "nhs.uk": {
        "name": "UK National Health Service",
        "trust_score": 1.0,
        "seeds": [
            "https://www.nhs.uk/conditions/diabetes/",
            "https://www.nhs.uk/conditions/type-1-diabetes/",
            "https://www.nhs.uk/conditions/type-2-diabetes/",
            "https://www.nhs.uk/live-well/eat-well/food-types/how-to-eat-a-balanced-diet/eating-a-balanced-diet/"
        ]
    },
    "webmd.com": {
        "name": "WebMD",
        "trust_score": 0.9,
        "seeds": [
            "https://www.webmd.com/diabetes/default.htm",
            "https://www.webmd.com/diabetes/guide/types-of-diabetes-mellitus",
            "https://www.webmd.com/diabetes/diabetes-diet-best-foods"
        ]
    },
    "clevelandclinic.org": {
        "name": "Cleveland Clinic",
        "trust_score": 1.0,
        "seeds": [
            "https://my.clevelandclinic.org/health/diseases/diabetes",
            "https://my.clevelandclinic.org/health/diseases/21501-type-1-diabetes",
            "https://my.clevelandclinic.org/health/diseases/21494-type-2-diabetes"
        ]
    },
    "hopkinsmedicine.org": {
        "name": "Johns Hopkins Medicine",
        "trust_score": 1.0,
        "seeds": [
            "https://www.hopkinsmedicine.org/health/conditions-and-diseases/diabetes",
            "https://www.hopkinsmedicine.org/health/wellness-and-prevention/diabetes-diet-best-foods-for-diabetes"
        ]
    }
}

# Topic classification keywords
t": 3, "managing": 3, "monitoring": 2}
    },
    "prevention": {
        "keywords": ["prevention", "preventing", "reduce risk", "prediabetes", "prevention program"],
        "weight": {"prevention": 3, "preventing": 3, "prediabetes": 2}
    }TOPIC_DEFINITIONS = {
    "overview": {
        "keywords": ["what is diabetes", "types of diabetes", "diabetes mellitus", "overview", "introduction", "basics"],
        "weight": {"what is": 3, "types": 2, "overview": 2}
    },
    "symptoms": {
        "keywords": ["symptoms", "signs", "warning signs", "early signs", "indicator", "manifestation", "frequent urination", "thirst", "fatigue"],
        "weight": {"symptoms": 3, "signs": 3, "warning": 2}
    },
    "causes": {
        "keywords": ["causes", "risk factors", "etiology", "pathophysiology", "insulin resistance", "genetics", "hereditary"],
        "weight": {"causes": 3, "risk factors": 3, "genetics": 2}
    },
    "diagnosis": {
        "keywords": ["diagnosis", "testing", "screening", "blood test", "A1C", "glucose test", "fasting glucose", "OGTT"],
        "weight": {"diagnosis": 3, "test": 2, "screening": 2}
    },
    "diet_nutrition": {
        "keywords": ["diet", "nutrition", "food", "eating", "meal plan", "carbohydrate", "sugar", "glycemic index", "healthy eating"],
        "weight": {"diet": 3, "nutrition": 3, "food": 2, "eating": 2}
    },
    "lifestyle": {
        "keywords": ["exercise", "physical activity", "lifestyle", "weight loss", "fitness", "active", "workout"],
        "weight": {"exercise": 3, "physical activity": 3, "lifestyle": 2}
    },
    "treatment": {
        "keywords": ["treatment", "medication", "insulin", "therapy", "metformin", "drug", "medicine", "injectable"],
        "weight": {"treatment": 3, "medication": 3, "insulin": 3}
    },
    "complications": {
        "keywords": ["complications", "damage", "neuropathy", "retinopathy", "nephropathy", "cardiovascular", "heart disease", "kidney disease", "foot problems"],
        "weight": {"complications": 3, "damage": 2, "neuropathy": 3}
    },
    "management": {
        "keywords": ["management", "managing diabetes", "blood sugar control", "glucose monitoring", "self-care", "diabetes care"],
        "weight": {"managemen
}

# -------------------------
# Data Models
# -------------------------
@dataclass
class ScrapedContent:
    url: str
    source_domain: str
    source_name: str
    title: str
    content: str
    topics: List[str]
    word_count: int
    quality_score: float
    trust_score: float
    scraped_date: str
    headers: List[str]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ScraperStats:
    total_attempted: int = 0
    successful: int = 0
    failed: int = 0
    blocked_robots: int = 0
    duplicate_content: int = 0
    low_quality: int = 0
    by_source: Dict[str, int] = None
    by_topic: Dict[str, int] = None
    
    def __post_init__(self):
        if self.by_source is None:
            self.by_source = {}
        if self.by_topic is None:
            self.by_topic = {}

# -------------------------
# Main Scraper Class
# -------------------------
class DiabetesKBScraper:
    def __init__(self, output_dir: str = "data", max_pages_per_source: int = 50, sources: dict = None):
        self.output_dir = output_dir
        self.max_pages_per_source = max_pages_per_source
        self.sources = sources if sources is not None else AUTHORITATIVE_SOURCES
        
        # Create directory structure
        self.raw_dir = os.path.join(output_dir, "raw")
        self.processed_dir = os.path.join(output_dir, "processed")
        self.topics_dir = os.path.join(output_dir, "topics")
        
        for directory in [self.raw_dir, self.processed_dir, self.topics_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # State management
        self.visited_urls: Set[str] = set()
        self.content_hashes: Set[str] = set()
        self.stats = ScraperStats()
        self.all_content: List[ScrapedContent] = []
        
        # Setup session
        self.session = self._create_session()
        self.robots_cache: Dict[str, robotparser.RobotFileParser] = {}
        self.domain_last_access: Dict[str, float] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Load progress if exists
        self._load_progress()
    
    def _setup_logging(self):
        log_file = os.path.join(self.output_dir, "scraper.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _load_progress(self):
        """Load previous scraping progress"""
        progress_file = os.path.join(self.output_dir, "progress.json")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    self.visited_urls = set(progress.get('visited_urls', []))
                    self.content_hashes = set(progress.get('content_hashes', []))
                logging.info(f"Resumed: {len(self.visited_urls)} URLs already processed")
            except Exception as e:
                logging.warning(f"Could not load progress: {e}")
    
    def _save_progress(self):
        """Save current progress"""
        progress_file = os.path.join(self.output_dir, "progress.json")
        progress = {
            'visited_urls': list(self.visited_urls),
            'content_hashes': list(self.content_hashes),
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if domain not in self.robots_cache:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            rp = robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robots_cache[domain] = rp
            except:
                self.robots_cache[domain] = None
        
        rp = self.robots_cache[domain]
        if rp is None:
            return True
        
        return rp.can_fetch("*", url)
    
    def _rate_limit(self, domain: str):
        """Enforce polite rate limiting per domain"""
        min_delay = 2.0  # Minimum 2 seconds between requests to same domain
        
        if domain in self.domain_last_access:
            elapsed = time.time() - self.domain_last_access[domain]
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        
        self.domain_last_access[domain] = time.time()
        
        # Additional random delay
        time.sleep(random.uniform(0.5, 1.5))
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication"""
        parsed = urlparse(url)
        # Remove fragments and common tracking params
        query_params = parse_qs(parsed.query)
        
        # Remove tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']
        for param in tracking_params:
            query_params.pop(param, None)
        
        # Rebuild query string
        from urllib.parse import urlencode
        clean_query = urlencode(query_params, doseq=True)
        
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{'?' + clean_query if clean_query else ''}"
    
    def _get_content_hash(self, text: str) -> str:
        """Generate hash for content deduplication"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title intelligently"""
        # Try multiple methods
        if soup.find('h1'):
            return soup.find('h1').get_text(strip=True)
        
        if soup.title:
            title = soup.title.get_text(strip=True)
            # Clean common title suffixes
            title = re.sub(r'\s*[\|\-]\s*(CDC|WHO|Mayo Clinic|WebMD|NHS).*$', '', title)
            return title
        
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content']
        
        return "Untitled"
    
    def _extract_content(self, soup: BeautifulSoup) -> Tuple[str, List[str]]:
        """Extract main content and headers"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find(class_=re.compile('content|article|main', re.I)) or
            soup.find(id=re.compile('content|article|main', re.I)) or
            soup.body
        )
        
        if not main_content:
            main_content = soup
        
        # Extract headers for structure
        headers = []
        for h in main_content.find_all(['h1', 'h2', 'h3']):
            header_text = h.get_text(strip=True)
            if header_text and len(header_text) > 3:
                headers.append(header_text)
        
        # Extract paragraphs and list items
        content_parts = []
        
        for elem in main_content.find_all(['p', 'li']):
            text = elem.get_text(separator=' ', strip=True)
            # Filter out very short fragments and navigation text
            if text and len(text.split()) > 8:
                content_parts.append(text)
        
        # Join content
        content = '\n\n'.join(content_parts)
        
        # Clean content
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\[\d+\]', '', content)  # Remove citation numbers
        content = re.sub(r'http\S+', '', content)  # Remove URLs from text
        
        return content.strip(), headers
    
    def _classify_topics(self, title: str, content: str, headers: List[str]) -> List[str]:
        """Classify content into relevant topics using keyword matching"""
        # Combine text for analysis (give more weight to title and headers)
        analysis_text = (
            title * 3 + ' ' + 
            ' '.join(headers) * 2 + ' ' + 
            content[:2000]
        ).lower()
        
        topic_scores = {}
        
        for topic, config in TOPIC_DEFINITIONS.items():
            score = 0
            for keyword in config['keywords']:
                count = analysis_text.count(keyword.lower())
                weight = config['weight'].get(keyword, 1)
                score += count * weight
            
            if score > 0:
                topic_scores[topic] = score
        
        # Return topics with scores above threshold, sorted by relevance
        threshold = 2
        relevant_topics = [
            topic for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold
        ]
        
        return relevant_topics[:3]  # Return top 3 topics
    
    def _calculate_quality_score(self, content: str, headers: List[str], soup: BeautifulSoup) -> float:
        """Calculate content quality score (0-1)"""
        score = 0.0
        word_count = len(content.split())
        
        # Word count scoring
        if word_count >= 100:
            score += 0.2
        if word_count >= 300:
            score += 0.2
        if word_count >= 500:
            score += 0.1
        
        # Structure scoring
        if len(headers) >= 2:
            score += 0.15
        if len(headers) >= 4:
            score += 0.1
        
        # Content depth indicators
        if soup.find_all(['ul', 'ol']):
            score += 0.1
        
        # Medical terminology (indicates authoritative content)
        medical_terms = ['insulin', 'glucose', 'blood sugar', 'A1C', 'diabetes mellitus', 
                        'pancreas', 'metabolism', 'hyperglycemia', 'hypoglycemia']
        term_count = sum(1 for term in medical_terms if term.lower() in content.lower())
        score += min(term_count * 0.05, 0.15)
        
        return min(score, 1.0)
    
    def _is_relevant_diabetes_content(self, content: str, title: str) -> bool:
        """Check if content is actually about diabetes"""
        diabetes_keywords = [
            'diabetes', 'diabetic', 'blood sugar', 'blood glucose', 'insulin',
            'hyperglycemia', 'hypoglycemia', 'A1C', 'glucose', 'prediabetes',
            'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes'
        ]
        
        combined_text = (title + ' ' + content[:1000]).lower()
        
        # Must have at least 2 diabetes-related terms
        matches = sum(1 for keyword in diabetes_keywords if keyword in combined_text)
        return matches >= 2
    
    def scrape_url(self, url: str, source_info: dict) -> Optional[ScrapedContent]:
        """Scrape a single URL"""
        # Normalize URL
        url = self._normalize_url(url)
        
        # Check if already visited
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        self.stats.total_attempted += 1
        
        # Check robots.txt
        if not self._check_robots_txt(url):
            logging.warning(f"Blocked by robots.txt: {url}")
            self.stats.blocked_robots += 1
            return None
        
        # Rate limiting
        domain = urlparse(url).netloc
        self._rate_limit(domain)
        
        # Make request
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content
            title = self._extract_title(soup)
            content, headers_list = self._extract_content(soup)
            
            # Validate content
            word_count = len(content.split())
            
            if word_count < 100:
                logging.info(f"Content too short ({word_count} words): {url}")
                self.stats.low_quality += 1
                return None
            
            # Check if content is about diabetes
            if not self._is_relevant_diabetes_content(content, title):
                logging.info(f"Content not relevant to diabetes: {url}")
                self.stats.low_quality += 1
                return None
            
            # Check for duplicate content
            content_hash = self._get_content_hash(content)
            if content_hash in self.content_hashes:
                logging.info(f"Duplicate content detected: {url}")
                self.stats.duplicate_content += 1
                return None
            
            self.content_hashes.add(content_hash)
            
            # Classify topics
            topics = self._classify_topics(title, content, headers_list)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(content, headers_list, soup)
            
            # Create scraped content object
            scraped_content = ScrapedContent(
                url=url,
                source_domain=domain,
                source_name=source_info['name'],
                title=title,
                content=content,
                topics=topics,
                word_count=word_count,
                quality_score=quality_score,
                trust_score=source_info['trust_score'],
                scraped_date=datetime.now().isoformat(),
                headers=headers_list
            )
            
            self.stats.successful += 1
            
            # Update stats
            self.stats.by_source[domain] = self.stats.by_source.get(domain, 0) + 1
            for topic in topics:
                self.stats.by_topic[topic] = self.stats.by_topic.get(topic, 0) + 1
            
            logging.info(f"✓ Scraped: {title[:60]}... ({word_count} words, Q:{quality_score:.2f}, Topics: {', '.join(topics)})")
            
            return scraped_content
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to scrape {url}: {e}")
            self.stats.failed += 1
            return None
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            self.stats.failed += 1
            return None
    
    def _discover_links(self, url: str, soup: BeautifulSoup, source_domain: str) -> List[str]:
        """Discover relevant links from a page"""
        discovered = []
        base_domain = urlparse(url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            # Parse URL
            parsed = urlparse(full_url)
            
            # Only follow links from the same trusted domain
            if parsed.netloc != base_domain:
                continue
            
            # Skip non-http(s) links
            if parsed.scheme not in ['http', 'https']:
                continue
            
            # Skip common non-content pages
            skip_patterns = [
                r'/search', r'/login', r'/signin', r'/signup', r'/register',
                r'/subscribe', r'/newsletter', r'/contact', r'/about',
                r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
                r'/video', r'/videos', r'/gallery', r'/images'
            ]
            
            if any(re.search(pattern, full_url, re.I) for pattern in skip_patterns):
                continue
            
            # Must contain diabetes-related keywords in URL or link text
            link_text = link.get_text(strip=True).lower()
            combined = (full_url + ' ' + link_text).lower()
            
            diabetes_keywords = ['diabetes', 'diabetic', 'blood-sugar', 'glucose', 'insulin', 'type-1', 'type-2']
            if any(keyword in combined for keyword in diabetes_keywords):
                discovered.append(full_url)
        
        return discovered
    
    def scrape_source(self, source_domain: str, source_info: dict):
        """Scrape all content from a specific source"""
        logging.info(f"\n{'='*80}")
        logging.info(f"Scraping: {source_info['name']} ({source_domain})")
        logging.info(f"{'='*80}")
        
        to_visit = list(source_info['seeds'])
        visited_this_source = set()
        scraped_count = 0
        
        while to_visit and scraped_count < self.max_pages_per_source:
            url = to_visit.pop(0)
            
            if url in visited_this_source:
                continue
            
            visited_this_source.add(url)
            
            # Scrape the URL
            content = self.scrape_url(url, source_info)
            
            if content:
                self.all_content.append(content)
                scraped_count += 1
                
                # Try to discover more links (only if we haven't hit the limit)
                if scraped_count < self.max_pages_per_source:
                    try:
                        response = self.session.get(url, timeout=15)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        new_links = self._discover_links(url, soup, source_domain)
                        
                        for link in new_links:
                            if link not in visited_this_source and link not in to_visit:
                                to_visit.append(link)
                    except:
                        pass
            
            # Save progress periodically
            if scraped_count % 5 == 0:
                self._save_progress()
        
        logging.info(f"Completed {source_info['name']}: {scraped_count} pages scraped")
    
    def save_results(self):
        """Save all scraped content in multiple formats"""
        logging.info("\nSaving results...")
        
        # 1. Save individual JSON files (for RAG ingestion)
        for idx, content in enumerate(self.all_content):
            filename = f"{content.source_domain}_{idx:04d}.json"
            filepath = os.path.join(self.processed_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 2. Save by topic (text files for easy reading)
        for topic in TOPIC_DEFINITIONS.keys():
            topic_content = [c for c in self.all_content if topic in c.topics]
            if topic_content:
                filepath = os.path.join(self.topics_dir, f"{topic}.txt")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"{'='*80}\n")
                    f.write(f"TOPIC: {topic.upper().replace('_', ' ')}\n")
                    f.write(f"Total Documents: {len(topic_content)}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    for content in topic_content:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Title: {content.title}\n")
                        f.write(f"Source: {content.source_name}\n")
                        f.write(f"URL: {content.url}\n")
                        f.write(f"Quality: {content.quality_score:.2f} | Trust: {content.trust_score:.2f}\n")
                        f.write(f"Topics: {', '.join(content.topics)}\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(content.content)
                        f.write("\n\n")
        
        # 3. Save master JSON with all content
        master_file = os.path.join(self.output_dir, "master_kb.json")
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in self.all_content], f, indent=2, ensure_ascii=False)
        
        # 4. Save statistics
        stats_file = os.path.join(self.output_dir, "scraping_stats.json")
        stats_dict = {
            'total_attempted': self.stats.total_attempted,
            'successful': self.stats.successful,
            'failed': self.stats.failed,
            'blocked_robots': self.stats.blocked_robots,
            'duplicate_content': self.stats.duplicate_content,
            'low_quality': self.stats.low_quality,
            'by_source': self.stats.by_source,
            'by_topic': self.stats.by_topic,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2)
        
        logging.info(f"✓ Saved {len(self.all_content)} documents to {self.output_dir}")
        logging.info(f"✓ Individual JSON files: {self.processed_dir}")
        logging.info(f"✓ Topic files: {self.topics_dir}")
        logging.info(f"✓ Master KB: {master_file}")
    
    def print_summary(self):
        """Print scraping summary"""
        print("\n" + "="*80)
        print("SCRAPING SUMMARY")
        print("="*80)
        print(f"Total URLs Attempted: {self.stats.total_attempted}")
        print(f"Successfully Scraped: {self.stats.successful}")
        print(f"Failed: {self.stats.failed}")
        print(f"Blocked by robots.txt: {self.stats.blocked_robots}")
        print(f"Duplicate Content: {self.stats.duplicate_content}")
        print(f"Low Quality/Irrelevant: {self.stats.low_quality}")
        
        print("\nContent by Source:")
        for source, count in sorted(self.stats.by_source.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} pages")
        
        print("\nContent by Topic:")
        for topic, count in sorted(self.stats.by_topic.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic.replace('_', ' ').title()}: {count} documents")
        
        print("\nQuality Metrics:")
        if self.all_content:
            avg_words = sum(c.word_count for c in self.all_content) / len(self.all_content)
            avg_quality = sum(c.quality_score for c in self.all_content) / len(self.all_content)
            avg_trust = sum(c.trust_score for c in self.all_content) / len(self.all_content)
            
            print(f"  Average Word Count: {avg_words:.0f}")
            print(f"  Average Quality Score: {avg_quality:.2f}")
            print(f"  Average Trust Score: {avg_trust:.2f}")
        
        print("="*80 + "\n")
    
    def run(self):
        """Main execution method"""
        logging.info("="*80)
        logging.info("DIABETES KNOWLEDGE BASE SCRAPER")
        logging.info("Scraping from genuine authoritative medical sources")
        logging.info("="*80)
        
        # Scrape each source
        for domain, source_info in self.sources.items():
            try:
                self.scrape_source(domain, source_info)
            except Exception as e:
                logging.error(f"Error scraping {domain}: {e}")
            
            # Save progress after each source
            self._save_progress()
        
        # Save all results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        logging.info("Scraping completed successfully!")

# -------------------------
# CLI Interface
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape diabetes knowledge base from authoritative medical sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diabetes_kb_scraper.py
  python diabetes_kb_scraper.py --output-dir ./diabetes_kb
  python diabetes_kb_scraper.py --max-pages 100 --output-dir ./data

Authoritative Sources:
  - CDC (Centers for Disease Control and Prevention)
  - WHO (World Health Organization)
  - Mayo Clinic
  - NIH/NIDDK (National Institute of Diabetes)
  - American Diabetes Association
  - NHS (UK National Health Service)
  - WebMD
  - Cleveland Clinic
  - Johns Hopkins Medicine
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='diabetes_kb_data',
        help='Output directory for scraped data (default: diabetes_kb_data)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=50,
        help='Maximum pages to scrape per source (default: 50)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start fresh'
    )
    
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=list(AUTHORITATIVE_SOURCES.keys()),
        help='Scrape only specific sources (default: all)'
    )
    
    args = parser.parse_args()
    
    # Reset if requested
    if args.reset:
        progress_file = os.path.join(args.output_dir, 'progress.json')
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"Reset progress file: {progress_file}")
    
    # Filter sources if specified
    sources_to_scrape = AUTHORITATIVE_SOURCES
    if args.sources:
        sources_to_scrape = {
            k: v for k, v in AUTHORITATIVE_SOURCES.items() 
            if k in args.sources
        }
        print(f"Scraping only: {', '.join(args.sources)}")
    
    # Run scraper with filtered sources
    scraper = DiabetesKBScraper(
        output_dir=args.output_dir,
        max_pages_per_source=args.max_pages,
        sources=sources_to_scrape
    )
    
    try:
        scraper.run()
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
        scraper._save_progress()
        scraper.save_results()
        scraper.print_summary()
        print("Progress saved. Run again to resume.")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        scraper._save_progress()
        print(f"\nError occurred: {e}")
        print("Progress saved. Check scraper.log for details.")

if __name__ == "__main__":
    main()
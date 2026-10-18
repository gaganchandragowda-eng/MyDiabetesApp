# 🏥 DiaCare - Diabetes Management & Prediction System

A modern web application for diabetes risk prediction using AI/ML, featuring separate portals for patients and doctors with a beautiful, colorful UI.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Credentials](#demo-credentials)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## ✨ Features

### For Patients
- 🔐 Secure registration and login
- 🔬 AI-powered diabetes risk prediction
- 📊 Interactive prediction form with helpful placeholders
- 📈 View prediction history
- 💾 Download PDF reports
- 📤 Upload lab reports for analysis
- 👤 Personal dashboard with statistics

### For Doctors
- 👨‍⚕️ Professional portal with green theme
- 📋 View patient predictions
- 📊 Analytics dashboard
- 🏥 Hospital/clinic management

### Design Features
- 🎨 Beautiful gradient designs (Purple for patients, Green for doctors)
- 📱 Fully responsive (mobile, tablet, desktop)
- ✨ Smooth animations and hover effects
- 🎯 User-friendly interface with icons
- 🔔 Flash message notifications

## 📁 Project Structure

```
DiaCare/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   ├── home.html                   # Landing page
│   ├── patient_login.html          # Patient login
│   ├── doctor_login.html           # Doctor login
│   ├── patient_register.html       # Patient registration
│   ├── doctor_register.html        # Doctor registration
│   ├── dashboard.html              # User dashboard
│   ├── predict.html                # Prediction form
│   ├── result.html                 # Results page
│   └── upload_lab.html             # Upload lab reports
│
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Main stylesheet
│   ├── images/
│   │   └── logo.svg               # DiaCare logo
│   └── js/
│       └── main.js                # JavaScript (optional)
│
└── models/                         # ML models
    └── diabetes_model.pkl         # Trained model (add your own)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone or Download
```bash
# If using git
git clone https://github.com/yourusername/diacare.git
cd diacare

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Project Files

1. **Create the templates folder** and save all HTML files:
   - home.html
   - patient_login.html
   - doctor_login.html
   - patient_register.html
   - doctor_register.html
   - dashboard.html
   - predict.html
   - result.html
   - upload_lab.html

2. **Create the static folder structure**:
```bash
mkdir -p static/css static/images static/js
```

3. **Save the CSS file** as `static/css/style.css`

4. **Save the logo** as `static/images/logo.svg`

5. **Create models folder** (for ML model):
```bash
mkdir models
```

### Step 5: Run the Application
```bash
python app.py
```

You should see:
```
🏥 DiaCare - Diabetes Management System
✨ Server starting...
🌐 Access the application at:
   - http://localhost:5000
   - http://127.0.0.1:5000
```

### Step 6: Open in Browser
Navigate to: **http://localhost:5000**

## 🎮 Usage

### First Time Setup

1. **Access the Home Page**
   - Open http://localhost:5000
   - You'll see the landing page with navigation

2. **Register an Account**
   - Click "Register" in navigation
   - Choose "Patient Register" or "Doctor Register"
   - Fill in the registration form
   - Submit to create account

3. **Login**
   - Click "Login" in navigation
   - Choose your user type
   - Enter email and password
   - You'll be redirected to dashboard

4. **Make a Prediction**
   - From dashboard, click "Predict Diabetes"
   - Fill in all health parameters
   - Each field has helpful placeholder text and info
   - Submit to get your prediction

5. **View Results**
   - See your risk probability with visual progress bar
   - Get personalized recommendations
   - Download PDF report (feature in development)

## 🔑 Demo Credentials

The application comes with demo accounts for testing:

### Patient Account
- **Email:** patient@demo.com
- **Password:** password

### Doctor Account
- **Email:** doctor@demo.com
- **Password:** password

## 📸 Screenshots

*(Add screenshots of your application here)*

### Home Page
![Home Page](screenshots/home.png)

### Login Pages
![Patient Login](screenshots/patient-login.png)
![Doctor Login](screenshots/doctor-login.png)

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Prediction Form
![Prediction Form](screenshots/predict.png)

### Results
![Results](screenshots/result.png)

## 🛠️ Technologies Used

### Backend
- **Flask** - Python web framework
- **Werkzeug** - Security and utilities
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning

### Frontend
- **HTML5** - Markup
- **CSS3** - Styling with gradients and animations
- **Bootstrap 5** - Responsive framework
- **Google Fonts** - Poppins font family
- **Vanilla JavaScript** - Interactivity

### Design
- **Color Scheme:**
  - Patient Portal: Purple gradient (#667eea → #764ba2)
  - Doctor Portal: Green gradient (#11998e → #38ef7d)
  - Background: Light gradients
- **Icons:** Emoji-based (can be replaced with Font Awesome)
- **Animations:** CSS keyframes for smooth transitions

## 📊 Adding Your ML Model

To use your own trained model:

1. Train your diabetes prediction model
2. Save it as pickle file:
```python
import pickle
pickle.dump(model, open('models/diabetes_model.pkl', 'wb'))
```

3. Uncomment this line in `app.py`:
```python
model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
```

The model should expect 8 features in this order:
1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🔒 Security Notes

**⚠️ For Production Deployment:**

1. **Change the Secret Key:**
```python
app.secret_key = 'your-very-secure-random-key-here'
```

2. **Use Environment Variables:**
```bash
# Create .env file
SECRET_KEY=your-secret-key
DATABASE_URL=your-database-url
```

3. **Use a Real Database:**
   - Replace dictionary storage with PostgreSQL/MySQL
   - Implement Flask-SQLAlchemy

4. **Enable HTTPS:**
   - Use SSL certificates
   - Configure secure cookies

5. **Add Input Validation:**
   - Server-side validation for all forms
   - Sanitize user inputs

6. **Implement Rate Limiting:**
   - Prevent brute force attacks
   - Use Flask-Limiter

## 🚀 Deployment

### Heroku Deployment

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Add to requirements.txt:
```
gunicorn==21.2.0
```

3. Deploy:
```bash
heroku create diacare-app
git push heroku main
```

### PythonAnywhere

1. Upload files via web interface
2. Configure WSGI file
3. Set working directory
4. Reload web app

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Bootstrap team for the framework
- Google Fonts for Poppins font
- Flask community for excellent documentation
- All contributors and testers

## 📞 Support

For support, email support@diacare.com or open an issue in the repository.

## 🗺️ Roadmap

- [ ] Add PDF report generation
- [ ] Implement email notifications
- [ ] Add data visualization charts
- [ ] Create admin dashboard
- [ ] Add appointment booking system
- [ ] Implement telemedicine features
- [ ] Add multi-language support
- [ ] Create mobile app version

---

**Made with ❤️ by DiaCare Team**
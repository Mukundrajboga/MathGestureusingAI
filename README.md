
# 📷 Math Gesture Solver

An AI-powered real-time Math Problem Solver that allows users to **draw mathematical equations in air using hand gestures** and get **instant AI-generated step-by-step solutions**, similar examples, voice feedback, and downloadable PDFs.

This project combines **Computer Vision, Hand Tracking, Generative AI, and Streamlit UI** to create an interactive learning system.

---

## 🚀 Features

* ✋ Hand gesture-based drawing using webcam
* 🎨 Custom drawing color selection
* 🧠 AI-powered math problem solving
* 📝 Step-by-step solution generation
* 🔁 Similar example generation with solution
* 🔊 Voice feedback for answers
* 📄 Auto PDF export of solutions
* 🖼 Export drawing in PDF, PNG, JPG
* ↩ Undo / ↪ Redo functionality
* 💾 Auto-save drawing history

---

## 🛠 Technologies Used

* Python
* OpenCV
* cvzone (Hand Tracking Module)
* NumPy
* Streamlit
* Google Gemini API
* Pillow (PIL)
* FPDF
* pyttsx3 (Text-to-Speech)

---

## 🧠 How It Works

1. Webcam captures live video.
2. Hand tracking detects finger positions.
3. User draws math equation in air using index finger.
4. When four fingers are raised, the system:

   * Captures the canvas
   * Sends it to Gemini AI
   * Generates:

     * Full solution
     * Step-by-step explanation
     * Similar solved example
5. Solution is:

   * Displayed in UI
   * Saved as PDF
   * Read aloud (if enabled)

---

## ✋ Gesture Controls

| Gesture         | Action           |
| --------------- | ---------------- |
| Index Finger Up | Draw             |
| Thumb Up        | Clear Canvas     |
| Four Fingers Up | Solve Problem    |
| Undo Button     | Undo Last Action |
| Redo Button     | Redo Action      |

---

## 📂 Project Structure

```
Math-Gesture-Solver/
│
├── app.py
├── requirements.txt
├── Saved_Answers/
├── Drawing_History/
└── README.md
```

---

## ⚙ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/math-gesture-solver.git
cd math-gesture-solver
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python cvzone numpy streamlit google-generativeai pillow fpdf pyttsx3
```

---

## 🔑 Setup Gemini API

Replace the API key inside the code:

```python
genai.configure(api_key="YOUR_API_KEY")
```

Get your API key from Google AI Studio.

---

## ▶ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

(Add screenshots of your UI here for better presentation)

---

## 🎯 Use Cases

* Students learning mathematics
* Interactive smart classroom tool
* Contactless equation solving
* AI-powered tutoring system
* Gesture-controlled educational application

---

## 💡 Future Improvements

* Multi-hand support
* Better equation recognition
* Cloud storage for saved history
* User login system
* Dark mode optimization
* Mobile camera integration

---

## 👨‍💻 Author

Mukund

---

## 📜 License

This project is for educational and research purposes.

---

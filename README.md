# URLGuard — Phishing URL Detector

A machine learning web app that detects phishing URLs by analyzing the text
of the URL itself (no need to visit the site). Paste a link in, and it tells
you whether it looks like phishing, how confident it is, and exactly which
red flags it found.

This guide assumes **zero coding experience**. Follow it top to bottom and
you'll have a working project on your computer, ready to put on your resume
and demo in an interview.

---

## 1. What this project actually does (read this first)

This project has just **3 code files**:

1. **`features.py`** — reads a URL as plain text and counts things like: how
   long is it, does it contain an IP address, does it say "verify" or
   "login", how many hyphens does it have, etc. This turns text into ~23
   numbers.
2. **`train_model.py`** — does everything needed to produce the trained
   model, in 3 clearly labeled steps inside the file: gathers real
   legitimate + real phishing URLs, converts them into features using
   `features.py`, then trains a **Random Forest** model (a standard,
   explainable machine learning algorithm) and saves it to
   `model/phishing_model.pkl`.
3. **`app.py`** — a small website (built with **Flask**, a Python web
   framework) where you paste a URL, and it runs the trained model on it
   live and shows you the verdict.

You will run these one at a time. Every file is heavily commented — open
them in a text editor and read through them once you're set up; that's
genuinely the fastest way to be able to explain this project confidently.

---

## 2. Install the tools you need (one-time setup)

### Step A — Install Python

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**
2. Click the big yellow "Download Python" button (get the latest version)
3. Run the installer.
   - **Windows:** on the very first install screen, check the box that says
     **"Add python.exe to PATH"** before clicking Install. This step is
     easy to miss and causes most beginner problems — don't skip it.
   - **Mac:** run the downloaded `.pkg` file and click through the installer.
4. Confirm it worked. Open a terminal:
   - **Windows:** press the Start key, type `cmd`, hit Enter.
   - **Mac:** press Cmd+Space, type `terminal`, hit Enter.
5. Type this and press Enter:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If Windows says
   `python` is not recognized, restart your computer and try again (PATH
   changes need a restart).
   > On some Macs you may need to type `python3` instead of `python`
   > everywhere in this guide.

### Step B — Install a code editor (optional but recommended)

Download **[VS Code](https://code.visualstudio.com/)** (free). This lets you
open the project folder, view files with color highlighting, and run
commands from a built-in terminal, all in one window.

### Step C — Get the project files

1. Unzip the project folder you downloaded from this chat somewhere easy to
   find, like your Desktop.
2. Open VS Code → File → Open Folder → select the unzipped `phishing_project`
   folder.
3. Open the built-in terminal in VS Code: menu **Terminal → New Terminal**.
   Every command below gets typed into that terminal.

   (No VS Code? Just open your normal terminal and `cd` into the folder,
   e.g. `cd Desktop/phishing_project`.)

---

## 3. Set up a virtual environment (keeps things clean)

A virtual environment is just an isolated folder for this project's Python
libraries, so they don't clash with anything else on your computer. Run:

```bash
python -m venv venv
```

Then **activate** it (you'll need to do this every time you reopen the
project in a new terminal session):

- **Windows (cmd):** `venv\Scripts\activate`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal line will now start with
`(venv)`.

---

## 4. Install the project's libraries

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```

This installs:
- **pandas** — handles the tabular data (the CSV files)
- **scikit-learn** — the machine learning library (trains the model)
- **joblib** — saves/loads the trained model to a file
- **Flask** — runs the local website

This might take a minute or two the first time.

---

## 5. Build the dataset and train the model

```bash
python train_model.py
```

This one script does it all: it gathers thousands of real legitimate and
phishing URLs, extracts features from each one (saving them to
`dataset.csv`), then trains the Random Forest model and prints its
accuracy, precision, recall, and F1 score on URLs it never saw during
training — that's how you know it's actually learning patterns, not just
memorizing. It saves the trained model to `model/phishing_model.pkl`.

You should see accuracy in the mid-to-high 90s%. **This step re-trains the
model from scratch — you don't strictly need to run it, since a
pre-trained model is already included, but running it yourself is good
practice and something you can screenshot for your portfolio.**

---

## 6. Run the web app

```bash
python app.py
```

You'll see something like:
```
Running on http://127.0.0.1:5000
```

Open that address in your web browser. Paste in a URL, click **Scan**, and
you'll get a verdict, a confidence percentage, and the specific red flags
found. Try the example links on the page to see both a phishing and a
legitimate result.

To stop the server, go back to the terminal and press `Ctrl+C`.

---

## 7. Understanding your results 

- **Accuracy / Precision / Recall / F1** shown at the bottom of the app are
  standard classification metrics:
  - *Accuracy* — % of all URLs classified correctly.
  - *Precision* — of the URLs flagged as phishing, what % actually were.
  - *Recall* — of all actual phishing URLs, what % did we catch.
  - *F1* — a balance between precision and recall.
- **Why Random Forest?** It's an ensemble of many decision trees voting
  together — accurate, resistant to overfitting compared to a single tree,
  and it gives you **feature importance** (which signals mattered most),
  which `train_model.py` prints out. This is a great thing to bring up in
  an interview: "the model relies most heavily on URL length, domain
  length, and HTTPS usage."
- **Known limitation:** URLs that are structurally ambiguous (e.g. a
  legitimate site with a hyphenated slug and a numeric ID in the path) can
  land close to the 50/50 decision boundary. This is a real, honest
  limitation of a lexical-features-only approach — production systems
  usually combine this with domain age, SSL certificate data, and
  reputation blocklists for a fuller picture. Mentioning this shows you
  understand the model's boundaries, not just its headline accuracy.

---

## 8. Ideas to extend this project (great "future work" talking points)

- Add domain-age / WHOIS lookup as an extra feature (requires a live
  network call per URL)
- Add a browser extension front-end instead of/alongside the web form
- Try other models (XGBoost, logistic regression) and compare
- Add a "report false positive" button that logs disagreements for review
- Deploy it publicly (e.g. Render, Railway, PythonAnywhere) so you can link
  a live demo on your resume

---

## Project structure

```
phishing_project/
├── features.py             # turns a URL into numeric features
├── train_model.py          # builds the dataset AND trains the model (all-in-one)
├── app.py                  # Flask web app
├── templates/
│   └── index.html          # web page UI
├── model/
│   ├── phishing_model.pkl  # pre-trained model (included)
│   └── metrics.json        # saved accuracy/precision/recall/F1
├── legit.csv               # raw legitimate URL list (source data)
├── phish.csv               # raw phishing URL list (source data)
├── dataset.csv             # generated training data (features + labels)
└── requirements.txt        # Python libraries needed
```

Just **3 files of actual code** (`features.py`, `train_model.py`, `app.py`)
plus one HTML page — everything else is data or generated output.

## Data sources & credit

- Phishing URLs: [PhishTank](https://phishtank.com) — a free, community
  phishing feed
- Legitimate URLs: University of New Brunswick benign URL list, combined
  with a curated list of well-known real-world domains


<img width="2227" height="1319" alt="Screenshot 2026-08-20 134408" src="https://github.com/user-attachments/assets/cf416ca3-6e44-4778-9711-75e8a92dcf01" />
<img width="2347" height="1212" alt="Screenshot 2026-08-20 134338" src="https://github.com/user-attachments/assets/d90ffbb6-a64b-453a-983d-43b57612ac66" />


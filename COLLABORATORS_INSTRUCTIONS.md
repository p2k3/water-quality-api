# Water Quality API & Dashboard: Collaborator Instructions

Welcome! This guide will help you set up, run, and demo the water quality prediction API and dashboard from scratch using VS Code. Follow each step carefully to avoid common issues.

---

## 1. Prerequisites
- **VS Code** installed
- **Git** installed
- **Python 3.8+** installed
- **Node.js (v18+) & npm** installed

---

## 2. Clone the Repository in VS Code
1. **Open VS Code**
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and type `Git: Clone`.
3. Enter the repository URL:
   ```
   https://github.com/p2k3/water-quality-api.git
   ```
4. Choose a local folder for the clone.
5. When prompted, click `Open` to open the cloned folder in VS Code.

---

## 3. Run the Backend (FastAPI)
1. **Open a new terminal in VS Code** (`Terminal > New Terminal`).
2. Navigate to the project root if not already there:
   ```powershell
   cd "c:\Users\Emmanuel's Laptop\Desktop\water-quality-api"
   ```
3. (Optional) Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. **Install Python dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. **Start the FastAPI server:**
   ```powershell
   uvicorn app.main:app --reload
   ```
   - The API will be available at `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`

---

## 4. Run the Frontend (React Dashboard)
1. **Open a second terminal in VS Code.**
2. Navigate to the frontend folder:
   ```powershell
   cd "app/water quality system/water-quality-frontend"
   ```
3. **Install frontend dependencies:**
   ```powershell
   npm install
   ```
4. **Start the React app:**
   ```powershell
   npm run dev
   ```
   - The dashboard will be available at `http://localhost:5173`

---

## 5. Demo Instructions
- **Submit water sample parameters** in the dashboard form.
- **View predictions, risk scores, parameter breakdown, pollutant probabilities, and explanations** in the dashboard.
- **API docs:** Visit `http://localhost:8000/docs` for interactive API testing.

---

## 6. Troubleshooting
- If you see errors about missing packages, re-run `pip install -r requirements.txt` or `npm install`.
- If ports are busy, stop other servers or change the port (e.g., `uvicorn app.main:app --reload --port 8001`).
- For Windows path issues, use double quotes around paths.
- For virtual environment issues, ensure you activate with `.\venv\Scripts\activate`.

---

## 7. Useful Commands
- **Stop a running server:** Press `Ctrl+C` in the terminal.
- **Reinstall dependencies:**
  - Backend: `pip install -r requirements.txt`
  - Frontend: `npm install`
- **View API docs:** `http://localhost:8000/docs`

---

## 8. Need Help?
- Check the README for more details.
- Contact the repo owner for support.

---

Happy collaborating!

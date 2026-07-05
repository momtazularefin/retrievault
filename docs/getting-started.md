# RetrieVault - Getting Started & Developer Operations

This document provides a comprehensive guide to setting up a clean developer machine from scratch, checking out the code, running the services, and managing local code changes.

---

## 1. Environment & Prerequisites Setup

To set up an empty laptop for RetrieVault development, install the following core tools:

### Git
* **Windows**: Download and install [Git for Windows](https://git-scm.com/download/win).
* **Mac/Linux**: Install via Homebrew (`brew install git`) or your system package manager.

### Docker Desktop
* Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
* **WSL2 Integration (Windows)**: Ensure WSL2 backend is enabled in Docker settings under *General -> Use the WSL 2 based engine*.

### Node.js (via NVM)
Using a Node Version Manager (NVM) allows you to manage node runtime environments cleanly:
1. **Windows**: Download and install [NVM-Windows](https://github.com/coreybutler/nvm-windows/releases).
2. **Execution**: Open your terminal and install Node v24.18.0:
   ```bash
   nvm install 24.18.0
   nvm use 24.18.0
   ```

### Astral `uv` (Python Dependency & Tool Manager)
Astral `uv` is the Python package and project manager used by RetrieVault. It handles local virtual environments and can **download the correct Python interpreter automatically**. You do NOT need to install Python globally on your machine.
* **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS/Linux**:
  ```bash
  curl -sSf https://astral.sh/uv/install.sh | sh
  ```

---

## 2. Code Checkout & Local Configuration

1. **Clone the repository**:
   ```bash
   git clone https://github.com/momtazularefin/retrievault.git
   cd retrievault
   ```
2. **Create local environment file**:
   Copy the provided environment template from the root folder:
   ```bash
   cp .env.example .env
   ```
   Open the newly created `.env` file and populate the required API keys (specifically `ANTHROPIC_API_KEY`). Refer to [docs/configuration.md](configuration.md) for full descriptions of settings.

---

## 3. Local Startup Workflow

To run the full-stack system locally, open separate terminal shells and execute these steps in order:

### Step 1: Start the Vector Store (Qdrant)
Run Qdrant via Docker Compose from the root directory:
```bash
docker compose up -d qdrant
```
* **Verify**: Visit the local Qdrant dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### Step 2: Initialize & Launch the Backend API
Navigate to the `backend` workspace, synchronize packages, and start the development server:
```bash
cd backend
uv sync --all-extras
uv run uvicorn retrievault.api:app --reload --host 0.0.0.0 --port 8000
```
* **Verify**: View the Swagger API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 3: Launch the Frontend Chat Interface
Navigate to the `frontend` workspace, install dependencies, and start the Next.js development server:
```bash
cd ../frontend
npm install
npm run dev
```
* **Verify**: Visit [http://localhost:3000](http://localhost:3000) to view the chat interface.

---

## 4. Change Management & Service Restarts

When modifying code during development, use the following rules to ensure changes are propagated correctly:

### Modifying Backend Python Code
* **Local Host Execution**: If you run Uvicorn on your host terminal (`uv run uvicorn ... --reload`), Uvicorn automatically detects file changes and reloads the API instantly.
* **Docker Compose Execution**: If you run the backend service inside Docker Compose (`docker compose up`), Windows file-system events occasionally fail to propagate to the container's volume mount. If your code edits do not reflect in the API:
  - Restart the container to reload the python process:
    ```bash
    docker compose restart backend
    ```
* **Verification**: Look at the glowing **Connected** status bar in the frontend footer. The `Build` hash and `Updated X seconds ago` timestamp will reset to indicate the new backend version is active.

### Modifying Frontend React/TypeScript Code
* The Next.js dev server uses **Fast Refresh**. Edits to React components, styling (`globals.css`), or page structures reflect in your browser **instantly** without reloading the page or restarting the npm server.

### Modifying Dependencies (Package Management)
* **Backend**: If you add packages to `backend/pyproject.toml` (e.g. using `uv add`):
  1. Synchronize the local environment: `uv sync --all-extras`
  2. If using Docker, rebuild the backend image layer:
     ```bash
     docker compose build backend
     docker compose up -d
     ```
* **Frontend**: If you add npm packages to `frontend/package.json`:
  1. Run `npm install` inside the `frontend/` directory.
  2. If the build server gets out-of-sync, stop the process (`CTRL+C`) and restart it: `npm run dev`.

---

## 5. Benchmarking & Verification

After implementing modifications, run the evaluation script to ensure retrieval accuracy, citation safety, and latency stay within bounds:
```bash
cd backend
uv run python -m retrievault.eval
```
The benchmark results and relative performance scores will be saved in `backend/eval/reports/report.md`. Refer to [docs/evaluation.md](evaluation.md) for grading metrics and custom heuristics.

---

## 6. Teardown & Cleanup Operations

To safely stop all running services and clean up temporary development files, follow these guidelines:

### Safe Shutdown Sequence

To avoid database write corruption or orphaned background processes, shut down services in the reverse order of startup:

1. **Next.js Frontend**: Stop the npm dev server by pressing `CTRL+C` in its terminal pane.
   * **Lingering Port Collision**: If you close the terminal but the port `3000` remains locked, a Node process is lingering in the background. Stop all active Node processes to release the port:
     * **Windows (PowerShell)**:
       ```powershell
       Stop-Process -Name node -Force -ErrorAction SilentlyContinue
       ```
     * **macOS/Linux**:
       ```bash
       killall node
       ```
2. **FastAPI Backend**: Stop the Uvicorn dev server by pressing `CTRL+C` in its terminal pane.
3. **Qdrant Vector Database**: Stop the database container from the root directory.
   
   > [!IMPORTANT]
   > Always stop the database gracefully using `docker compose stop` or `docker compose down`. This sends a `SIGTERM` signal to Qdrant, allowing it to flush its in-memory Write-Ahead Log (WAL) and index segments to disk safely. Avoid force-killing the container (`docker kill`), which can lead to index corruption.

   * **Stop only the Database (Keep other containers running)**:
     ```bash
     docker compose stop qdrant
     ```
   * **Full Tear Down (Stop & remove all containers, preserve index data)**:
     ```bash
     docker compose down
     ```
   * **Complete Clean Wipe (Stop, remove containers, and delete vector index)**: Use this if you want to completely wipe the Qdrant database volume to test a fresh codebase ingestion:
     ```bash
     docker compose down -v
     ```

### Cleaning Temporary Cache Files

The development environment and evaluation harness generate local caches. You can clean these up using the following commands:

* **Evaluation Caches (Layer 1 & Layer 2)**: 
  Delete the evaluation response cache and the Ragas LLM judge cache to force grading from scratch:
  * **Windows (PowerShell)**:
    ```powershell
    Remove-Item backend/eval/.cache_responses.json, backend/eval/.langchain_cache.db -ErrorAction SilentlyContinue
    ```
  * **macOS/Linux**:
    ```bash
    rm -f backend/eval/.cache_responses.json backend/eval/.langchain_cache.db
    ```
* **Python Runtime Caches**:
  Clear all compiled `__pycache__` folders and unit test run caches:
  * **Windows (PowerShell)**:
    ```powershell
    Get-ChildItem -Path . -Filter __pycache__ -Recurse | Remove-Item -Recurse -Force
    Remove-Item .pytest_cache, backend/.pytest_cache -Recurse -Force -ErrorAction SilentlyContinue
    ```
  * **macOS/Linux**:
    ```bash
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .pytest_cache backend/.pytest_cache
    ```
* **Next.js Cache**:
  Delete the local compilation folder `.next` to clear hot-reload build caches and resolve hydration glitches:
  * **Windows (PowerShell)**:
    ```powershell
    Remove-Item frontend/.next -Recurse -Force -ErrorAction SilentlyContinue
    ```
  * **macOS/Linux**:
    ```bash
    rm -rf frontend/.next
    ```

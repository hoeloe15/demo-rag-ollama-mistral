import subprocess

def run_backend():
    """Run the backend server."""
    print("Starting backend server...")
    backend_process = subprocess.Popen(["python", "backend.py"])
    return backend_process

def run_frontend():
    """Run the frontend server."""
    print("Starting frontend server...")
    frontend_process = subprocess.Popen(["streamlit", "run", "frontend/app.py"])
    return frontend_process

def main():
    backend_process = run_backend()

    try:
        # Wait for the backend process to complete
        backend_process.wait()
    except KeyboardInterrupt:
        print("Backend interrupted. Shutting down...")
        backend_process.terminate()
        backend_process.wait()
        return

    # Run frontend after backend has completed
    frontend_process = run_frontend()

    try:
        # Wait for the frontend process to complete
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Frontend interrupted. Shutting down...")
        frontend_process.terminate()
        frontend_process.wait()

if __name__ == "__main__":
    main()

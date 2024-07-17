import subprocess
import time

def run_backend():
    """Run the backend server."""
    print("Starting backend server...")
    backend_process = subprocess.Popen(["python", "backend.py"])
    return backend_process

def run_frontend():
    """Run the frontend server."""
    print("Starting frontend server...")
    frontend_command = ["streamlit", "run", "app.py", "local"]
    frontend_process = subprocess.Popen(frontend_command)
    return frontend_process

def main():
    backend_process = run_backend()

    # Wait for a few seconds to ensure the backend is fully started
    time.sleep(10)

    # Run frontend after waiting
    frontend_process = run_frontend()

    try:
        # Wait for the frontend process to complete
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Frontend interrupted. Shutting down...")
        frontend_process.terminate()
        frontend_process.wait()

    # Ensure backend is also terminated if frontend is interrupted
    backend_process.terminate()
    backend_process.wait()

if __name__ == "__main__":
    main()

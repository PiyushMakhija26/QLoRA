import subprocess
import sys


def main() -> None:
    # Run streamlit run command using subprocess
    cmd = ["streamlit", "run", "src/invoice_extractor/ui/app.py"]
    print(f"Launching Streamlit Dashboard: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception as e:
        print(f"Error launching Streamlit server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

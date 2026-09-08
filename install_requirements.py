import sys
import subprocess

def install_requirements():
    print("=" * 60)
    print("📦 Menginstall Seluruh Library Dependency yang Dibutuhkan...")
    print("=" * 60)
    
    req_file = "requirements.txt"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("\n✅ Installasi berhasil!")
    except Exception as e:
        print(f"\n❌ Gagal menginstall dependency: {e}")

if __name__ == "__main__":
    install_requirements()

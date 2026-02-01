import sys
import os

def check_python_version():
    print("=" * 60)
    print("1. Checking Python version...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ❌ ERROR: Python 3.10+ required")
        return False
    if version.minor < 11:
        print("   ⚠️  WARNING: Python 3.11+ recommended, but 3.10 will work")
    print("   ✅ Python version OK")
    return True

def check_dependencies():
    print("\n2. Checking Python dependencies...")
    required = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pyodbc",
        "aioodbc",
        "pydantic",
        "pydantic_settings",
        "jose",
    ]
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n   ERROR: Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True

def check_odbc_driver():
    print("\n3. Checking ODBC Driver...")
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        sql_drivers = [d for d in drivers if "SQL Server" in d]
        if sql_drivers:
            print(f"   ✅ Found SQL Server drivers: {', '.join(sql_drivers)}")
            return True
        else:
            print("   ❌ ERROR: No SQL Server ODBC driver found")
            print("   Install: Microsoft ODBC Driver 17 for SQL Server")
            return False
    except ImportError:
        print("   ❌ ERROR: pyodbc not installed")
        return False

def check_env_file():
    print("\n4. Checking environment configuration...")
    if os.path.exists(".env"):
        print("   ✅ .env file exists")
        return True
    else:
        print("   ⚠️  WARNING: .env file not found")
        print("   Creating from env.example...")
        if os.path.exists("env.example"):
            try:
                import shutil
                shutil.copy("env.example", ".env")
                print("   ✅ Created .env from env.example")
                return True
            except Exception as e:
                print(f"   ❌ Failed to create .env: {e}")
                return False
        else:
            print("   ❌ env.example not found")
            return False

def check_database_connection():
    print("\n5. Checking database connection...")
    try:
        from app.core.config import settings
        print(f"   Database: {settings.DB_NAME}")
        print(f"   Server: {settings.DB_SERVER}")
        print(f"   Driver: {settings.DB_DRIVER}")
        
        import asyncio
        from app.db.database import init_db
        
        try:
            asyncio.run(init_db())
            print("   ✅ Database connection successful")
            return True
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            print("\n   Troubleshooting:")
            print("   - Check SQL Server is running")
            print("   - Verify database name in .env")
            print("   - Check Windows Authentication")
            return False
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def check_imports():
    print("\n6. Checking application imports...")
    try:
        from app.core.config import settings
        from app.core.logging import setup_logging
        from app.api.v1.router import api_router
        from app.middleware.security import SecurityMiddleware
        from app.middleware.error_handler import ErrorHandlerMiddleware
        from app.db.database import init_db
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("ExpenseAlly AI Service - Setup Diagnostic")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_odbc_driver(),
        check_env_file(),
        check_imports(),
        check_database_connection(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All checks passed! You can run the app with: python main.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 60 + "\n")
    
    return all(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


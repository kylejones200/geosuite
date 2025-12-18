# Backup of original Dash app - moved to app_dash_backup.py
# Now using Flask with blueprint structure

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

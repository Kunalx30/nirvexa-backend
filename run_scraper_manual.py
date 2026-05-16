from app import create_app
from app.services.scheduler import _app, run_daily_job_pipeline
import app.services.scheduler as scheduler_module

def main():
    print("Initializing App...")
    app = create_app()
    
    with app.app_context():
        print("Setting app context for scheduler...")
        # Inject the app so the scheduler has the context it needs
        scheduler_module._app = app
        
        print("Running the manual job pipeline. This may take a few minutes as it scrapes multiple sources...")
        run_daily_job_pipeline()
        print("Pipeline execution complete!")

if __name__ == "__main__":
    main()

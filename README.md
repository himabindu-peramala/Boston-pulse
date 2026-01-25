# Boston Pulse

Boston Pulse is a machine learning–driven digital twin of the City of Boston. It unifies municipal open data, real-time transit and weather feeds, and community sentiment into a single conversational system that helps residents and newcomers understand, navigate, and make decisions about city life. Instead of interacting with fragmented city portals and dashboards, Boston Pulse synthesizes civic, safety, housing, and mobility data into actionable, context-aware insights delivered through a natural language interface.

## Project Objectives

- Integrate heterogeneous Boston city datasets into a unified City State
- Build predictive models for civic services, neighborhood recommendations, and urban risk
- Enable natural language interaction with structured and real-time city data
- Demonstrate an end-to-end data and machine learning pipeline grounded in public data

## Data Sources

The project primarily relies on datasets from Analyze Boston, including:

- 311 Service Requests
- Crime Incident Reports
- Fire Incident Reporting
- Public Works Violations
- RentSmart Housing Data
- Street Address Management (SAM) and Street Segments
- Bluebike Stations

Additional real-time data sources include MBTA alerts and weather APIs.

## Repository Structure

- `data/` – Raw, processed, and feature-engineered datasets  
- `notebooks/` – Exploratory analysis and prototyping  
- `src/` – Data ingestion, preprocessing, feature engineering, and modeling code  
- `config/` – Configuration files and parameters  
- `app/` – Application and conversational interface  

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/boston-pulse.git
cd boston-pulse
Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run data ingestion scripts to fetch raw datasets from Analyze Boston and external APIs. Execute preprocessing and feature engineering pipelines to construct the City State. Train predictive models using scripts in the `src/models` directory. Launch the application to interact with Boston Pulse. Detailed steps and examples are documented in the project notebooks.

## Notes

- Raw datasets are not fully stored in the repository due to size constraints.  
- API keys and sensitive configurations should be stored in environment variables or configuration files excluded via `.gitignore`.  
- Some datasets update frequently; model outputs reflect the data available at run time.


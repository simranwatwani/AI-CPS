import pandas as pd
import requests
import json

def load_data_from_github():
    
    print("🔍 Loading data...")
    
    api_url = "https://api.github.com/repos/simranwatwani/AI-CPS/contents/data/raw"
    
    try:
        # Make request to GitHub API
        response = requests.get(api_url)
        if response.status_code != 200:
            raise Exception(f"Failed to access GitHub API: {response.status_code}")
        
        
        items = json.loads(response.text)
        
        # Search for CSV files
        for item in items:
            if item["name"].endswith(".csv"):
                raw_url = item["download_url"]
                print(f"Found dataset file: {item['name']}")
                print(f"RAW URL: {raw_url}")
                
                # Load the CSV directly from GitHub
                df = pd.read_csv(raw_url, encoding="latin1")
                print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
        
        raise FileNotFoundError("No CSV file found in GitHub repository.")
        
    except Exception as e:
        print(f"GitHub scraping failed: {e}")
        raise

def load_data_from_kaggle():
    
    print("Note: Using GitHub instead of Kaggle...")
    return load_data_from_github()

if __name__ == "__main__":
    print("Testing GitHub scraper...")
    df = load_data_from_github()
    print(df.head())
import os
import sys
import requests
from tqdm import tqdm
import hashlib

def download_from_release(repo_owner, repo_name, release_tag, asset_name, destination, expected_md5=None):
    """Download a file from a GitHub release"""
    url = f"https://github.com/{repo_owner}/{repo_name}/releases/download/{release_tag}/{asset_name}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists and has correct hash
    if os.path.exists(destination) and expected_md5:
        print(f"Checking if existing file at {destination} matches expected MD5...")
        with open(destination, 'rb') as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()
        if file_md5 == expected_md5:
            print(f"File already exists with correct MD5 hash")
            return True
        else:
            print(f"Existing file has different MD5 hash, will download again")
    
    # Download the file
    try:
        print(f"Downloading {asset_name} from GitHub release...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB chunks for faster download
        
        print(f"Total download size: {total_size / (1024 * 1024):.2f} MB")
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        
        # Verify MD5 hash if provided
        if expected_md5:
            print("Verifying file integrity...")
            with open(destination, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != expected_md5:
                print(f"Warning: MD5 hash mismatch. Expected {expected_md5}, got {file_md5}")
                return False
            print("File integrity verified!")
        
        print(f"Download complete: {destination}")
        return True
    
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        if os.path.exists(destination):
            print(f"Removing incomplete download file")
            os.remove(destination)
        return False

def download_all_assets():
    """Download all required assets from GitHub release"""
    # Repository details for the download
    repo_owner = "alejandrohdez00"
    repo_name = "art-evolution"
    release_tag = "v1.0.0"
    
    # Define all assets to download
    assets = [
        {
            "name": "embeddings.npy",
            "destination": "assets/embedding/embeddings.npy",
            "md5": None
        },
        {
            "name": "best_model_resnet_152.pth",
            "destination": "assets/embedding/best_model_resnet_152.pth",
            "md5": None
        }
        # Add more assets as needed
    ]
    
    print("=" * 80)
    print(f"Downloading assets from GitHub Releases")
    print(f"Repository: {repo_owner}/{repo_name}")
    print(f"Release: {release_tag}")
    print(f"Release URL: https://github.com/{repo_owner}/{repo_name}/releases/tag/{release_tag}")
    print("=" * 80)
    
    all_success = True
    
    for asset in assets:
        print(f"\nDownloading {asset['name']}...")
        success = download_from_release(
            repo_owner, 
            repo_name, 
            release_tag, 
            asset['name'], 
            asset['destination'], 
            asset['md5']
        )
        
        if not success:
            all_success = False
            print(f"❌ Failed to download {asset['name']}")
    
    if all_success:
        print("\n✅ All assets successfully downloaded and ready to use!")
        print(f"You can now run art evolution with originality fitness function")
    else:
        print("\n❌ Some assets failed to download.")
        print(f"Please download manually from: https://github.com/{repo_owner}/{repo_name}/releases/tag/{release_tag}")
        sys.exit(1)

def main():
    download_all_assets()

if __name__ == "__main__":
    main() 
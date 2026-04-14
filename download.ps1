# dataset link https://zenodo.org/records/14622048/files/dfc25_track1_trainval.zip?download=1
# proably easier to just enter that link into your browser

# Create directories
New-Item -Path . -ItemType Directory -Name dataset -Force | Out-Null
New-Item -Path .\dataset -ItemType Directory -Name openearthmap-sar -Force | Out-Null

# Download the file
$uri = 'https://zenodo.org/records/14622048/files/dfc25_track1_trainval.zip?download=1'
$output = '.\dataset\openearthmap-sar\download.zip'
Invoke-WebRequest -Uri $uri -OutFile $output

# Change to the download directory and unzip
Set-Location -Path '.\dataset\openearthmap-sar'
Expand-Archive -LiteralPath 'download.zip' -DestinationPath '.' -Force
Set-Location -Path '../../'
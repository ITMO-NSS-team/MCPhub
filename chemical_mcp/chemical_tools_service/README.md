# Chemical Tools Service

Service for different chemical tools.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Minimum 8GB RAM

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/RodionGolovinsky/OpenChemIE_service
cd OpenChemIE_service
```

2. Start the service:
```bash
docker compose up
```

The service will be available at: `http://localhost:8000`

## API Endpoints

### Main endpoints

- `GET /` - Service information
- `POST /extract_reactions_from_pdf/` - Extract reactions from PDF
- `POST /extract_reactions_from_figure/` - Extract reactions from image
- `POST /extract_molecules_from_pdf/` - Extract molecules from PDF
- `POST /extract_molecules_from_figure/` - Extract molecules from image
- `POST /convert_image_to_smiles/` - Convert image to SMILES
- `POST /docking/` - Perform molecular docking

### Usage examples

#### Extract reactions from PDF
```bash
curl -X POST "http://localhost:8000/extract_reactions_from_pdf/" \
     -H "Content-Type: multipart/form-data" \
     -F "pdf_file=@document.pdf"
```

#### Extract molecules from image
```bash
curl -X POST "http://localhost:8000/extract_molecules_from_figure/" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@molecule.png"
```

#### Convert to SMILES
```bash
curl -X POST "http://localhost:8000/convert_image_to_smiles/" \
     -H "Content-Type: multipart/form-data" \
     -F "image_file=@structure.png"
```

#### Perform molecular docking
```bash
curl -X POST "http://localhost:8000/docking/?smiles=CCO&pdb_id=1hsg"
```

The docking endpoint performs molecular docking of a ligand (specified as SMILES) to a protein (specified by PDB ID). It returns:
- **affinity**: Binding affinity results from docking
- **visualization**: Base64-encoded HTML visualization of the docking results

## Technical Details

### Used models
- **OpenChemIE** - for extracting chemical reactions and molecules
- **MolScribe** - for converting images to SMILES
- **Smina** - for molecular docking calculations

### Supported formats
- **PDF**: .pdf files
- **Images**: .jpg, .jpeg, .png

### System requirements
- **CPU**: x86_64
- **GPU**: NVIDIA GPU with CUDA 12.2+ (optional)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk**: 10GB free space


from rdkit import Chem
from rdkit.Chem import AllChem
from docking.docking_py.docking_py.docking import Docking
import os
import py3Dmol
import logging
import base64
from typing import Dict, Optional
from pdb_manip_py import pdb_manip

logging.basicConfig(level=logging.INFO)


def visualize_docking(protein_pdb: str, ligand_pdb: str, output_file: str = "docking_view.html"):
    """
    Visualize the docking results.
    
    Arguments:
        protein_pdb: path to protein PDB file
        ligand_pdb: path to ligand PDB file
        output_file: path to output HTML file (default: "docking_view.html")
    
    Returns:
        str: path to output HTML file on success, None on failure
    """
    try:
        if not os.path.exists(protein_pdb):
            logging.error(f"Protein PDB file not found: {protein_pdb}")
            return None
        
        if not os.path.exists(ligand_pdb):
            logging.error(f"Ligand PDB file not found: {ligand_pdb}")
            return None
        
        viewer = py3Dmol.view(width=600, height=400)
        
        try:
            with open(protein_pdb, 'r') as f:
                protein = f.read()
        except IOError as e:
            logging.error(f"Failed to read protein PDB file {protein_pdb}: {str(e)}")
            return None

        try:
            with open(ligand_pdb, 'r') as f:
                ligand = f.read()
        except IOError as e:
            logging.error(f"Failed to read ligand PDB file {ligand_pdb}: {str(e)}")
            return None

        try:
            viewer.addModel(protein, "pdb")
            viewer.setStyle({"cartoon": {"color": "spectrum"}})

            viewer.addModel(ligand, "pdbqt")
            viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon"}})

            viewer.zoomTo()
            html = viewer._make_html()
        except Exception as e:
            logging.error(f"Failed to create 3D visualization: {str(e)}")
            return None

        try:
            with open(output_file, "w") as f:
                f.write(html)
            logging.info(f"Visualization saved to {output_file}")
            return output_file
        except IOError as e:
            logging.error(f"Failed to write output file {output_file}: {str(e)}")
            return None
    
    except Exception as e:
        logging.error(f"Visualization failed with error: {str(e)}", exc_info=True)
        return None


def run_docking(smiles: str, pdb_id: str, cpu: int = 4) -> Dict:
    """
    Full cycle: SMILES → 3D-ligand → loading protein → preparation → docking → output affinities.
    
    Arguments:
        smiles: SMILES-line of the ligand (e.g. 'CCO')
        pdb_id: ID protein from PDB (e.g. '1hsg')
        cpu: number of CPU cores for docking (default 4)
    
    Returns:
        Dictionary with keys:
            - success: bool - whether docking was successful
            - affinity: dict/None - affinity results from docking
            - out_pdb_path: str - path to output PDB file
            - message: str - status message
            - error: str/None - error message if failed
            - visualization: str/None - base64 encoded HTML visualization
    """
    try:
        os.makedirs("data", exist_ok=True)
        logging.info(f"Preparing ligand from SMILES: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "success": False,
                "affinity": None,
                "out_pdb_path": None,
                "message": "Failed to parse SMILES string",
                "error": "Invalid SMILES format"
            }
        
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            return {
                "success": False,
                "affinity": None,
                "out_pdb_path": None,
                "message": "Failed to generate 3D conformer",
                "error": "Could not embed molecule in 3D space"
            }
        
        AllChem.UFFOptimizeMolecule(mol)
        lig_pdb = f"data/{pdb_id}_ligand.pdb"
        Chem.MolToPDBFile(mol, lig_pdb)

        logging.info(f"Loading protein {pdb_id} from PDB...")
        coor = pdb_manip.Coor()
        coor.get_PDB(pdb_id, f"data/{pdb_id}.pdb")

        rec_coor = coor.select_part_dict(selec_dict={'res_name': pdb_manip.PROTEIN_RES})
        rec_pdb = f"./data/{pdb_id}_rec.pdb"
        rec_coor.write_pdb(rec_pdb)
        
        dock = Docking(name=f"dock_{pdb_id}", rec_pdb=rec_pdb, lig_pdb=lig_pdb)
        dock.prepare_receptor(check_file_out=False)
        dock.prepare_ligand(check_file_out=False)

        out_pdb = f"./data/{pdb_id}_out.pdb"
        dock.run_docking(
            out_pdb=out_pdb,
            dock_bin="smina",
            num_modes=10,
            energy_range=5,
            exhaustiveness=8,
            cpu=cpu,
            autobox=True,
            check_file_out=False
        )
        aff = dock.affinity
        
        visualization_html = None
        viz_file = f"data/{pdb_id}_docking_view.html"
        if os.path.exists(out_pdb) and os.path.exists(rec_pdb):
            viz_result = visualize_docking(rec_pdb, out_pdb, output_file=viz_file)
            if viz_result and os.path.exists(viz_file):
                try:
                    with open(viz_file, "rb") as f:
                        html_content = f.read()
                    visualization_html = base64.b64encode(html_content).decode('utf-8')
                    logging.info("Visualization created successfully")
                except Exception as e:
                    logging.warning(f"Failed to encode visualization HTML: {str(e)}")
                    visualization_html = None
            else:
                logging.warning("Visualization creation failed or file not found")
        else:
            logging.warning(f"PDB files not found for visualization: rec={rec_pdb}, out={out_pdb}")
        
        if aff is None:
            logging.warning("Docking completed but affinity results are missing.")
            return {
                "success": False,
                "affinity": None,
                "message": "Docking completed but affinity extraction failed",
                "error": "Affinity results not available",
                "visualization": visualization_html
            }
        else:
            return {
                "success": True,
                "affinity": aff,
                "message": "Docking completed successfully",
                "error": None,
                "visualization": visualization_html
            }
    
    except Exception as e:
        logging.error(f"Docking failed with error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "affinity": None,
            "message": "Docking process failed",
            "error": str(e),
            "visualization": None
        }
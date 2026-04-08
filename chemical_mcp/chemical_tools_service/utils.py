from PIL import Image, ImageDraw
from typing import Dict, Any, List


def _clean_molecule_data(molecule: Dict[str, Any], include_molfile: bool = False) -> Dict[str, Any]:
    """Removes unnecessary fields from molecule description."""
    keys_to_remove = {'category_id', 'category'}
    if not include_molfile:
        keys_to_remove |= {'molfile', 'atoms', 'bonds'}
    return {k: v for k, v in molecule.items() if k not in keys_to_remove}


def _clean_reaction_entities(entities: List[Dict[str, Any]]):
    """Cleans the list of reaction entities (reactants, products, conditions)."""
    for ent in entities:
        for field in ('category_id', 'category', 'molfile', 'atoms', 'bonds'):
            ent.pop(field, None)



def visualize_reactions(results):
    for index, result in enumerate(results):
        print(result)
        image = result['figure']
        image.save(f"image_original_{index}.png")
        draw = ImageDraw.Draw(image)
        
        for reaction in result['reactions']:
            reactants = reaction['reactants']
            for reactant in reactants:
                x1, y1, x2, y2 = reactant['bbox']
                x1 = int(x1 * image.width)
                y1 = int(y1 * image.height)
                x2 = int(x2 * image.width)
                y2 = int(y2 * image.height)
                draw.rectangle((x1-5, y1-5, x2+5, y2+5), outline='blue', width=2)
                draw.text((x1-5, y1-5), 'reactant', fill='blue')
            conditions = reaction['conditions']
            for condition in conditions:
                x1, y1, x2, y2 = condition['bbox']
                x1 = int(x1 * image.width)
                y1 = int(y1 * image.height)
                x2 = int(x2 * image.width)
                y2 = int(y2 * image.height)
                draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
                draw.text((x1-5, y1-5), 'condition', fill='red')
            products = reaction['products']
            for product in products:
                x1, y1, x2, y2 = product['bbox']
                x1 = int(x1 * image.width)
                y1 = int(y1 * image.height)
                x2 = int(x2 * image.width)
                y2 = int(y2 * image.height)
                draw.rectangle((x1, y1, x2, y2), outline='green', width=2)
                draw.text((x1+5, y1+5), 'product', fill='green')
        image.save(f"image_annotated_{index}.png")
        with open(f"results_annotated_{index}.txt", "w") as f:
            f.write(str(result))
            print(f"Результаты сохранены в results_{index}.json")
            

def visualize_molecules(results):
    for index, result in enumerate(results):
        image = Image.fromarray(result['image'])
        image.save(f"image_original_{index}.png")
        draw = ImageDraw.Draw(image)
        
        for molecule in result['reactions']:
            x1, y1, x2, y2 = molecule['bbox']
            x1 = int(x1 * image.width)
            y1 = int(y1 * image.height)
            x2 = int(x2 * image.width)
            y2 = int(y2 * image.height)
            draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
        image.save(f"image_annotated_{index}.png")
        with open(f"results_annotated_{index}.txt", "w") as f:
            f.write(str(result))
            print(f"Результаты сохранены в results_{index}.json")
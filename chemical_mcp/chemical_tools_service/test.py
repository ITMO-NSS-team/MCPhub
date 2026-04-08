import requests
import base64
import os

# Параметры для запроса
url = "http://localhost:8000/docking/"
smiles = "CCO"  # Пример SMILES (этанол)
pdb_id = "1hsg"  # Пример PDB ID

# Параметры запроса
params = {
    "smiles": smiles,
    "pdb_id": pdb_id
}

response = requests.post(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    if data.get("success") and data.get("data", {}).get("visualization"):
        html_base64 = data["data"]["visualization"]
        html_content = base64.b64decode(html_base64)
        
        output_file = f"docking_result_{pdb_id}.html"
        with open(output_file, "wb") as f:
            f.write(html_content)
        
        print(f"HTML файл успешно сохранен: {output_file}")
        print(f"Affinity: {data.get('data', {}).get('affinity')}")
    else:
        print(f"Ошибка: {data.get('message', 'Unknown error')}")
        if data.get("error"):
            print(f"Детали ошибки: {data['error']}")
else:
    print(f"HTTP ошибка: {response.status_code}")
    print(f"Ответ: {response.text}")


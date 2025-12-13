import os
import pickle

import requests

# token = os.environ.get('TOKEN')
token = "1234_qwer"

output_dir = "temp/se"


def unstructure_by_service(doc_path, url, token):
    with open(doc_path, "rb") as f:
        base_name = os.path.basename(doc_path)
        # 提取文件名中第一个"_"前面的三位数字
        if "_" in base_name:
            base_name = base_name.split("_")[0]
        pickle_path = os.path.join(output_dir, f"{base_name}.pkl")

        files = {"file": f}
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(url, files=files, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        result = response_data.get("result")

        with open(pickle_path, "wb") as pkl_file:
            pickle.dump(result, pkl_file)


dir_path = "data/se"

# pdf_url = 'http://localhost:8770/pdf'
pdf_url = "http://thuenv.tiangong.world:7770/mineru"
# pdf_url = 'http://192.168.8.1:7770/mineru'
# docx_url = 'http://localhost:8770/docx'
docx_url = "http://thuenv.tiangong.world:7770/docx"
# ppt_url = 'http://localhost:8770/ppt'

# doc_ls = os.listdir(dir_path)
doc_ls = ["062_GB-T 34664-2017《电子电气生态设计产品评价通则》.PDF"]

# print(doc_ls)

for doc in doc_ls:
    if doc.lower().endswith(".pdf"):
        doc_path = os.path.join(dir_path, doc)
        unstructure_by_service(doc_path, pdf_url, token)
    elif doc.lower().endswith(".docx"):
        doc_path = os.path.join(dir_path, doc)
        unstructure_by_service(doc_path, docx_url, token)
    # elif doc.endswith('.pptx'):
    #     doc_path = os.path.join(dir_path, doc)
    #     unstructure_by_service(doc_path, ppt_url, token)

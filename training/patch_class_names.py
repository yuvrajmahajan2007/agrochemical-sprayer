import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "datasets", "processed")
MODELDIR = os.path.join(ROOT, "models")

import train_model as t
m = t.load_manifest()
cl = t.build_class_list(m)
split_train = os.path.join(ROOT, "datasets", "split", "train")
folders = sorted(os.listdir(split_train))
class_info = {}
for i, folder in enumerate(folders):
    entry = next((c for c in cl if c["folder"] == folder), None)
    if entry is None:
        print("MISSING", folder)
        continue
    if entry["status"] == "Healthy":
        display = "{} Healthy".format(entry["crop"])
    elif entry["crop"].lower() in entry["display_name"].lower():
        display = entry["display_name"]
    else:
        display = "{} {}".format(entry["crop"], entry["display_name"])
    class_info[str(i)] = {
        "folder_name": folder,
        "crop": entry["crop"],
        "display_name": display,
        "disease": entry["display_name"] if entry["status"] != "Healthy" else None,
        "status": entry["status"],
        "source": entry["source"],
    }
with open(os.path.join(MODELDIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_info, f, indent=2, ensure_ascii=False)
print("Rewrote class_names.json with", len(class_info), "classes")
for i in [0, 1, 2, 3, 11, 32]:
    print(i, class_info[str(i)]["display_name"])
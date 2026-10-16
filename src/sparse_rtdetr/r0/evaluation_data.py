"""548-development only; separate identity from the historical 64 trace reader."""
import io,json,hashlib
from pathlib import Path
from PIL import Image
from ..localization.data import validate_role_roots,read_bound,parse_raw,RawAugmentation
from ..localization.material_paths import MANIFESTS,metadata_path
class Development548:
 def __init__(self,spec):
  self.spec=spec
  validate_role_roots("development",spec["roots"]["images"],spec["roots"]["annotations"],spec["roots"])
  path=metadata_path("development",spec["manifest"],MANIFESTS["development"][0]);raw=read_bound(path,MANIFESTS["development"][0],path.stat().st_size)
  self.records=json.loads(raw)["records"]
  if len(self.records)!=548 or len({r["stable_image_id"] for r in self.records})!=548 or any(r["split"]!="val" for r in self.records):raise ValueError("R0 fixed development membership")
  self.transform=RawAugmentation((768,1344));self.by_id={r["stable_image_id"]:r for r in self.records}
 def raw(self,row):
  if self.by_id.get(row.get("stable_image_id"))!=row:raise ValueError("R0 unknown evaluation member")
  validate_role_roots("development",self.spec["roots"]["images"],self.spec["roots"]["annotations"],self.spec["roots"])
  raw=read_bound(Path(self.spec["roots"]["annotations"])/Path(row["annotation_relative_path"]).name,row["annotation_sha256"],row["annotation_size_bytes"])
  parsed=parse_raw(raw,row["stable_image_id"],expected_rows=row["raw_annotation_rows"])
  if parsed["counts"].get("nonpositive_area",0):raise ValueError("unsupported nonpositive official GT")
  return parsed
 def item(self,row,position):
  raw=self.raw(row);b=read_bound(Path(self.spec["roots"]["images"])/Path(row["relative_path"]).name,row["image_sha256"],row["image_size_bytes"])
  with Image.open(io.BytesIO(b)) as im:
   im=im.convert("RGB")
   if (im.width,im.height)!=(row["width"],row["height"]):raise ValueError("development dimensions mismatch")
   pixels,_,trace=self.transform(im,raw,epoch=120,seed=0,position=position,image_id=row["stable_image_id"],training=False)
  if trace["geometry_steps"]:raise ValueError("evaluation must be resize only; no augmentation")
  return pixels,raw


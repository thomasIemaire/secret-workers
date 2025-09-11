from bson import ObjectId
from datetime import datetime
from typing import Literal, Optional
import os, time, random, re, copy

context_db = None

def run_task(*, doc: dict= None, db=None, MAX_WORKERS=2) -> dict:
    if doc is None: return 
    
    global context_db
    context_db = db
    
    col_tasks = db.get_collection("datasets")
    col_data = db.get_collection("datasets_data")
    col_models = db.get_collection("models")
    col_configs = db.get_collection("models_configurations")

    docdtid = str(doc["_id"])
    model = col_models.find_one({"_id": ObjectId(doc.get("model"))})

    mversion = model.get("version", "1.0")
    ments = model.get("entities", {})
    mkeys = list(ments.keys())

    mcid = model.get("configuration", None)
    if not mcid:
        raise ValueError("Model configuration is missing")

    configuration = col_configs.find_one({"_id": ObjectId(mcid)})

    dataset = []

    size = doc.get("size", "recommended")
    n_max = configuration.get("possibilities", 1e5)
    n_size = size.get("size", n_max)
    if is_integer(n_size):
        n_size = int(n_size)
    else:
        n_size = model_build_calculate_size(n_size, n_max, len(configuration.get("formats", [])))

    progress = 0
    col_tasks.update_one({"_id": ObjectId(docdtid)}, {"$set": {"status": "generating", "progress": progress}})

    for _ in range(n_size):
        mvb = build_model_configuration(copy.deepcopy(configuration))

        try:
            mrd = random.choice(model.get("randomizers", []))
            rdm = build_model_configuration_randomizers(mrd)
            mvb["format"] = rdm(mvb["format"])
        except: pass
        
        dataset.append(
            build_model_entity(
                mvb,
                mkeys
            )
        )

        time.sleep(1e-9)

        n_progress = float((len(dataset) / n_size))
        if n_progress > progress:
            progress = n_progress
            col_tasks.update_one({"_id": ObjectId(docdtid)}, {"$set": {"progress": progress}})

    for data in dataset:
        col_data.insert_one({
            "dataset": ObjectId(docdtid),
            "data": data,
            "created_at": datetime.utcnow(),
        })

    col_tasks.update_one({"_id": ObjectId(docdtid)}, {"$set": {"status": "generated", "progress": 0.0}})

def model_build_calculate_size(
        size: str,
        max_size: int,
        formats_size: int
    ) -> int:
    match size:
        case "complete":
            return max_size
        case "advanced":
            return max_size // 2
        case "recommended":
            return max_size // formats_size
        case "small":
            return max_size // formats_size // 2
        case "tiny":
            return max_size // formats_size // 5
        case _:
            return int(1e3)

def build_model_configuration(
        configuration: dict
    ) -> dict:
    catt = configuration.get("attributes")
    cfmt = configuration.get("formats")

    sfmt = random.choice(cfmt)
    satt = []

    for attr in catt:
        kattr = attr.get("key")
        fattr = attr.get("frequency", 1)
        rattr = attr.get("requirements", [])
        vattr = attr.get("value") if fattr > random.random() else False

        if isinstance(vattr, dict):
            tvattr = vattr.get("type")
            rvattr = vattr.get("rule")
            pvattr = vattr.get("parameters", {})
            bvattr, bcattrs = build_model_configuration_value(tvattr, rvattr, pvattr)

            if bcattrs:
                for bcattr in bcattrs:
                    satt.append(bcattr)
        
        satt.append({
            "key": kattr,
            "value": bvattr if vattr else '',
            "requirements": build_model_configuration_requirements(bvattr, rattr) if vattr else True
        })

    bfmt = build_model_configuration_format(sfmt, satt)
    configuration['attributes'] = satt
    configuration['format'] = re.sub(r'\s+', ' ', bfmt.strip())

    return configuration

def build_model_configuration_value(
    vtype: Literal["number", "string"],
    rule: str,
    parameters: dict
) -> Optional[int | str]:
    value = None

    match rule:
        case "randint":
            vmin = int(parameters.get("min", 0))
            vmax = int(parameters.get("max", 100))
            if vmin > vmax: vmin, vmax = vmax, vmin
            value = random.randint(vmin, vmax)

        case "data":
            data_id = parameters.get("object_id")
            if data_id:
                data = context_db.get_collection("models_data").find_one({"_id": ObjectId(data_id)})
                value = random.choice(data.get("data", []))

        case "configuration":
            config_id = parameters.get("object_id")
            if config_id:
                config = context_db.get_collection("models_configurations").find_one({"_id": ObjectId(config_id)})
                value = build_model_configuration(config)
                return value.get("format", ""), config.get("attributes", [])
    
    if value is None: return None, None
        
    return build_model_configuration_vtype(vtype, value), None

def build_model_configuration_vtype(
    vtype: Literal["number", "string"],
    value: any
) -> Optional[int | str]:
    match vtype:
        case "number":
            try:
                return int(value)
            except:
                return str(value)
        case _:
            return str(value)

def build_model_configuration_requirements(
    value: any,
    requirements: list[dict]
) -> bool:
    for req in requirements:
        rreq = req.get("rule")
        creq = req.get("constraint")

        match rreq:
            case "regex":
                if not re.match(creq, str(value)):
                    return False
            case "eq":
                if str(value) != str(creq):
                    return False
            case "neq":
                if str(value) == str(creq):
                    return False
            case "gt":
                try:
                    if float(value) <= float(creq):
                        return False
                except: pass
            case "lt":
                try:
                    if float(value) >= float(creq):
                        return False
                except: pass
            case "gte":
                try:
                    if float(value) < float(creq):
                        return False
                except: pass
            case "lte":
                try:
                    if float(value) > float(creq):
                        return False
                except: pass
            case "in":
                creq  = [x.strip() for x in creq.split(",")] if isinstance(creq, str) else creq
                if str(value) not in map(str, creq):
                    return False
            case "nin":
                creq  = [x.strip() for x in creq.split(",")] if isinstance(creq, str) else creq
                if str(value) in map(str, creq):
                    return False
            case "contains":
                if str(creq) not in str(value):
                    return False
            case "ncontains":
                if str(creq) in str(value):
                    return False
            case _:
                pass
                
    return True

def build_model_configuration_format(
    format: str,
    attributes: list[dict]
) -> str:
    for attr in attributes:
        kattr = attr.get("key")
        vattr = attr.get("value", "")
        format = format.replace(f"{{{kattr}}}", str(vattr))
    return format

def build_model_configuration_randomizers(
    randomizer:  str
) -> lambda x: x:
    rrand = randomizer.get("rule")
    frand = randomizer.get("frequency", 1)

    f = None

    match rrand:
        case "upper":
            f = lambda x: x.upper()
        case "lower":
            f = lambda x: x.lower()
        case _:
            f = lambda x: x

    return f if frand >= random.random() else lambda x: x

def build_model_entity(
    configuration: dict,
    keys: list[str],
) -> dict:
    vfmt = configuration.get("format", "")
    ents = []

    for attr in configuration.get("attributes", []):
        kattr = attr.get("key")
        vattr = attr.get("value", "")
        rattr = attr.get("requirements", True)

        if not kattr in keys or \
            vattr == '' or not rattr:
            continue

        strvattr = str(vattr)
        sta = vfmt.lower().find(strvattr.lower()) if vattr else -1
        if sta == -1: continue
        end = sta + len(strvattr)

        ents.append([sta, end, kattr])
    
    return { "text": vfmt, "entities": ents }

def is_integer(value: any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False
    
def bump_version(version: str, bump: str) -> str:
    major, minor = map(int, version.split("."))
    if bump == "major":
        return f"{major + 1}.0"
    return f"{major}.{minor + 1}"
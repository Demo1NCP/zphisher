import string,binascii,os,random,secrets,hashlib,base64,cv2,requests
import numpy as np
from MedoSigner import Argus,Gorgon,Ladon
from time import time
from uuid import uuid4
from flask import *
from fastapi import FastAPI
from pydantic import BaseModel
class PuzzleSolver:
    def __init__(self, base64puzzle, base64piece):
        self.puzzle = base64puzzle
        self.piece = base64piece
        self.methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED
        ]

    def get_position(self):
        try:
            results = []

            puzzle = self.__background_preprocessing()
            piece = self.__piece_preprocessing()

            for method in self.methods:
                matched = cv2.matchTemplate(puzzle, piece, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
                if method == cv2.TM_SQDIFF_NORMED:
                    results.append((min_loc[0], 1 - min_val))
                else:
                    results.append((max_loc[0], max_val))

            enhanced_puzzle = self.__enhanced_preprocessing(puzzle)
            enhanced_piece = self.__enhanced_preprocessing(piece)

            for method in self.methods:
                matched = cv2.matchTemplate(enhanced_puzzle, enhanced_piece, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
                if method == cv2.TM_SQDIFF_NORMED:
                    results.append((min_loc[0], 1 - min_val))
                else:
                    results.append((max_loc[0], max_val))

            edge_puzzle = self.__edge_detection(puzzle)
            edge_piece = self.__edge_detection(piece)

            matched = cv2.matchTemplate(edge_puzzle, edge_piece, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
            results.append((max_loc[0], max_val))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[0][0]

        except Exception as e:
            puzzle = self.__background_preprocessing()
            piece = self.__piece_preprocessing()
            matched = cv2.matchTemplate(puzzle, piece, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
            return max_loc[0]

    def __background_preprocessing(self):
        img = self.__img_to_array(self.piece)
        background = self.__sobel_operator(img)
        return background

    def __piece_preprocessing(self):
        img = self.__img_to_array(self.puzzle)
        template = self.__sobel_operator(img)
        return template

    def __enhanced_preprocessing(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        return enhanced

    def __edge_detection(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def __sobel_operator(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)

        return grad

    def __img_to_array(self, base64_input):
        try:
            img_data = base64.b64decode(base64_input)
            img_array = np.frombuffer(img_data, dtype=np.uint8)

            decoded_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            if decoded_img is None:
                raise ValueError("Failed to decode image")

            if len(decoded_img.shape) == 2:
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_GRAY2BGR)
            elif decoded_img.shape[2] == 4:
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGBA2BGR)

            return decoded_img

        except Exception as e:
            raise ValueError(f"Image processing error: {str(e)}")


def sign(params: str, payload: str  = None, sec_device_id: str = '', cookie: str = None, aid: int = 1233, license_id: int = 1611921764, sdk_version_str: str = 'v05.00.06-ov-android', sdk_version: int = 167775296, platform: int = 0, unix: float = None):
    x_ss_stub = hashlib.md5(payload.encode('utf-8')).hexdigest() if payload != None else None
    if not unix: unix = time()

    return Gorgon(params, unix, payload, cookie).get_value() | {
        'content-length' : str(len(payload)),
        'x-ss-stub'      : x_ss_stub.upper(),
        'x-ladon'        : Ladon.encrypt(int(unix), license_id, aid),
        'x-argus'        : Argus.get_sign(params, x_ss_stub, int(unix),
            platform        = platform,
            aid             = aid,
            license_id      = license_id,
            sec_device_id   = sec_device_id,
            sdk_version     = sdk_version_str, 
            sdk_version_int = sdk_version
        )
    }


class CaptchaSolver:
    def __init__(self, iid: str, did: str, device_type: str, device_brand: str, host:str,proxy: str = None):
        self.iid = iid
        self.did = did
        self.device_type = device_type
        self.device_brand = device_brand

        self.host = host
        self.host_region = self.host.split('-')[2].split('.')[0]
        self.country = 'au'
        if proxy:
            self.session = requests.Session()
            self.session.proxies = {
                "http": f"http://{proxy}",
                "https": f"http://{proxy}"
            }
        else:
            self.session = requests.Session()

    def get_captcha(self):
        params = f'lang=en&app_name=musical_ly&version_name=32.9.5&h5_sdk_version=2.33.7&h5_sdk_use_type=cdn&sdk_version=2.3.4.i18n&iid={self.iid}&did={self.did}&device_id={self.did}&ch=googleplay&aid=1233&os_type=0&mode=slide&tmp={int(time())}{random.randint(111, 999)}&platform=app&webdriver=undefined&verify_host=https%3A%2F%2F{self.host_region}%2F&locale=en&channel=googleplay&app_key&vc=32.9.5&app_version=32.9.5&session_id&region={self.host_region}&use_native_report=1&use_jsb_request=1&orientation=2&resolution=720*1280&os_version=25&device_brand={self.device_brand}&device_model={self.device_type}&os_name=Android&version_code=3275&device_type={self.device_type}&device_platform=Android&type=verify&detail=&server_sdk_env=&imagex_domain&subtype=slide&challenge_code=99996&triggered_region={self.host_region}&cookie_enabled=true&screen_width=360&screen_height=640&browser_language=en&browser_platform=Linux%20i686&browser_name=Mozilla&browser_version=5.0%20%28Linux%3B%20Android%207.1.2%3B%20{self.device_type}%20Build%2FN2G48C%3B%20wv%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Version%2F4.0%20Chrome%2F86.0.4240.198%20Mobile%20Safari%2F537.36%20BytedanceWebview%2Fd8a21c6'
        sig = sign(params, '', "AadCFwpTyztA5j9L" + ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(9)), None, 1233)
        headers = {
            'X-Tt-Request-Tag': 'n=1;t=0',
            'X-Vc-Bdturing-Sdk-Version': '2.3.4.i18n',
            'X-Tt-Bypass-Dp': '1',
            'Content-Type': 'application/json; charset=utf-8',
            'X-Tt-Dm-Status': 'login=0;ct=0;rt=7',
            'X-Tt-Store-Region': self.country,
            'X-Tt-Store-Region-Src': 'did',
            'User-Agent': f'com.zhiliaoapp.musically/2023209050 (Linux; U; Android 7.1.2; en_{self.country.upper()}; {self.device_type}; Build/N2G48C;tt-ok/3.12.13.4-tiktok)',
            "x-ss-req-ticket": sig["x-ss-req-ticket"],
            "x-ss-stub": sig["x-ss-stub"],
            "X-Gorgon": sig["x-gorgon"],
            "X-Khronos": str(sig["x-khronos"]),
            "X-Ladon": sig["x-ladon"],
            "X-Argus": sig["x-argus"]
        }

        response = self.session.get(
            f'https://{self.host}/captcha/get?{params}',
            headers=headers,
        ).json()

        return response

    def verify_captcha(self, data):
        params = f'lang=en&app_name=musical_ly&version_name=32.9.5&h5_sdk_version=2.33.7&h5_sdk_use_type=cdn&sdk_version=2.3.4.i18n&iid={self.iid}&did={self.did}&device_id={self.did}&ch=googleplay&aid=1233&os_type=0&mode=slide&tmp={int(time())}{random.randint(111, 999)}&platform=app&webdriver=undefined&verify_host=https%3A%2F%2F{self.host}%2F&locale=en&channel=googleplay&app_key&vc=32.9.5&app_version=32.9.5&session_id&region={self.host_region}&use_native_report=1&use_jsb_request=1&orientation=2&resolution=720*1280&os_version=25&device_brand={self.device_brand}&device_model={self.device_type}&os_name=Android&version_code=3275&device_type={self.device_type}&device_platform=Android&type=verify&detail=&server_sdk_env=&imagex_domain&subtype=slide&challenge_code=99996&triggered_region={self.host_region}&cookie_enabled=true&screen_width=360&screen_height=640&browser_language=en&browser_platform=Linux%20i686&browser_name=Mozilla&browser_version=5.0%20%28Linux%3B%20Android%207.1.2%3B%20{self.device_type}%20Build%2FN2G48C%3B%20wv%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Version%2F4.0%20Chrome%2F86.0.4240.198%20Mobile%20Safari%2F537.36%20BytedanceWebview%2Fd8a21c6'
        sig = sign(params, '', "AadCFwpTyztA5j9L" + ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(9)), None, 1233)
        headers = {
            'X-Tt-Request-Tag': 'n=1;t=0',
            'X-Vc-Bdturing-Sdk-Version': '2.3.4.i18n',
            'X-Tt-Bypass-Dp': '1',
            'Content-Type': 'application/json; charset=utf-8',
            'X-Tt-Dm-Status': 'login=0;ct=0;rt=7',
            'X-Tt-Store-Region': self.country,
            'X-Tt-Store-Region-Src': 'did',
            'User-Agent': f'com.zhiliaoapp.musically/2023209050 (Linux; U; Android 7.1.2; en_{self.country.upper()}; {self.device_type}; Build/N2G48C;tt-ok/3.12.13.4-tiktok)',
            "x-ss-req-ticket": sig["x-ss-req-ticket"],
            "x-ss-stub": sig["x-ss-stub"],
            "X-Gorgon": sig["x-gorgon"],
            "X-Khronos": str(sig["x-khronos"]),
            "X-Ladon": sig["x-ladon"],
            "X-Argus": sig["x-argus"]
        }

        response = self.session.post(
            f'https://{self.host}/captcha/verify?{params}',
            headers=headers,
            json=data,
        ).text
        return response

    def start(self) -> None:
        
            _captcha = self.get_captcha()

            captcha_data = _captcha["data"]["challenges"][0]

            captcha_id = captcha_data["id"]
            verify_id = _captcha["data"]["verify_id"]

            puzzle_img = self.session.get(captcha_data["question"]["url1"]).content
            piece_img = self.session.get(captcha_data["question"]["url2"]).content

            puzzle_b64 = base64.b64encode(puzzle_img)
            piece_b64 = base64.b64encode(piece_img)

            solver = PuzzleSolver(puzzle_b64, piece_b64)
            max_loc = solver.get_position()

            rand_length = random.randint(50, 100)
            movements = []

            for i in range(rand_length):
                progress = (i + 1) / rand_length
                x_pos = round(max_loc * progress)

                y_offset = random.randint(-2, 2) if i > 0 and i < rand_length - 1 else 0
                y_pos = captcha_data["question"]["tip_y"] + y_offset

                movements.append({
                    "relative_time": i * rand_length + random.randint(-5, 5),
                    "x": x_pos,
                    "y": y_pos
                })

            verify_payload = {
                "modified_img_width": 552,
                "id": captcha_id,
                "mode": "slide",
                "reply": movements,
                "verify_id": verify_id
            }

            return self.verify_captcha(verify_payload)


def afriton(host,ht: str = None):
    
        if not ht:ht=None
        os_version = f"{random.randint(7, 33)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        iid      = int(bin(int(time()) + random.randint(0, 100))[2:] + "10100110110100110000011100000101", 2)
        did      = int(bin(int(time()) + random.randint(0, 100))[2:] + "00101101010100010100011000000110", 2)
        samsung = ["SM-G975F","SM-G532G","SM-N975F","SM-G988U","SM-G977U","SM-A705FN","SM-A515U1","SM-G955F","SM-A750G","SM-N960F","SM-G960U","SM-J600F","SM-A908B","SM-A705GM","SM-G970U","SM-A307FN","SM-G965U1","SM-A217F","SM-G986B","SM-A207M","SM-A515W","SM-A505G","SM-A315G","SM-A507FN","SM-A505U1","SM-G977T","SM-A025G","SM-J320F","SM-A715W","SM-A908N","SM-A205F","SM-G988B","SM-N986B","SM-A715F","SM-A515F","SM-G965F","SM-G960F","SM-A505F","SM-A207F","SM-A307G","SM-G970F","SM-A107F","SM-G935F","SM-G935A","SM-A310F","SM-J320FN"]
        oppo =['CPH2359','CPH2457','CPH2349','CPH2145','CPH2293','CPH2343','CPH2127','CPH2197','CPH2173','CPH2371','CPH2269','CPH2005','CPH2185']
        realme=['RMX3501','RMX3085','RMX1921','RMX3771','RMX3461','RMX3092','RMX3393','RMX3392','RMX1821','RMX1825','RMX3310',]
        phone=random.choice([samsung,oppo,realme])
        type1=random.choice(phone)
        if 'SM' in type1 :
                brand='samsung'
                dev=type1.split('-')[1]
        if 'RMX' in type1:
                brand='realme'
                dev=type1.split('X')[1]
        if 'CPH' in type1:
                brand='OPPO'
                dev=type1.split('H')[1]
        openudid = str(binascii.hexlify(os.urandom(8)).decode())
        device_type= type1
        device_brand= brand
        os_version= os_version
        cdid= str(uuid4())
        try:
            captcha_status = CaptchaSolver(iid,did,device_type,device_brand,host,ht).start()
            if "Verification complete" in  captcha_status:
                return iid,did,openudid,os_version,cdid,device_type,device_brand
        except:return "ER"

app = FastAPI()
class Item(BaseModel):
    
    device: str
    prox: str
@app.post("/AFRIT/CaptchaSolver")
async def create_item(item: Item):
  
    host=item.device
    prox=item.prox


    iid,did,openudid,os_version,cdid,device_type,device_brand=afriton(host,prox)
    
    return {"AFRIT":"Tle:@AFR_0 | @US_SB","iid":iid  ,
            "iid":iid  ,
            "did":did  ,
            "openudid":openudid  ,
            "os_version":os_version  ,
            "cdid":cdid  ,
            "device_type":device_type  ,
            "device_brand":device_brand }

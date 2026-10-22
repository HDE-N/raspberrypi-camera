import time
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


class CameraManager:
    """管理多顆樹莓派 CSI 相機或 USB 相機的硬體切換、影像擷取與前處理。"""

    def __init__(
        self,
        camera_num: int = 0,
        width: int = 1920,
        height: int = 1080,
        rotate_180: bool = True,
        swap_rb: bool = True,
    ):
        """初始化相機管理器。

        Args:
            camera_num (int): 硬體鏡頭編號 (例如 0 代表 CAM0, 1 代表 CAM1)，預設 0。
            width (int): 影像寬度 (單位: pixels)，預設 1920。
            height (int): 影像高度 (單位: pixels)，預設 1080。
            rotate_180 (bool): 是否將影像旋轉 180 度，預設 True。
            swap_rb (bool): 是否對調 Red/Blue 色彩通道，預設 True。
        """
        self.camera_num = camera_num
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180
        self.swap_rb = swap_rb
        self.picam2 = None
        self.cap = None
        self.is_connected = False
        self.error_msg = ""
        self.init_hardware(camera_num, width, height)

    @staticmethod
    def list_available_cameras() -> list[dict]:
        """查詢系統目前已連接的硬體相機清單。

        Args:
            無。

        Returns:
            list[dict]: 相機資訊列表，每個元素包含 'num' (int) 與 'model' (str)。
        """
        cameras = []
        if HAS_PICAMERA2:
            try:
                info_list = Picamera2.global_camera_info()
                for idx, info in enumerate(info_list):
                    cameras.append(
                        {
                            "num": info.get("Num", idx),
                            "model": info.get("Model", f"CSI_Cam_{idx}"),
                        }
                    )
            except Exception:
                pass
        return cameras

    def init_hardware(
        self, camera_num: int, width: int, height: int
    ) -> tuple[bool, str]:
        """安全釋放舊相機並根據指定的鏡頭編號與解析度重新啟動硬體。

        Args:
            camera_num (int): 目標硬體鏡頭編號 (0, 1, ...)。
            width (int): 目標影像寬度 (單位: pixels)。
            height (int): 目標影像高度 (單位: pixels)。

        Returns:
            tuple[bool, str]:
                - success (bool): 硬體啟動成功回傳 True，失敗回傳 False。
                - message (str): 狀態或錯誤說明訊息。
        """
        self.release()
        self.camera_num = int(camera_num)
        self.width = int(width)
        self.height = int(height)

        if HAS_PICAMERA2:
            try:
                self.picam2 = Picamera2(camera_num=self.camera_num)
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.is_connected = True
                self.error_msg = ""
                return True, f"成功啟動 CSI 鏡頭 #{self.camera_num} ({self.width}x{self.height})"
            except Exception as e:
                self.picam2 = None
                self.is_connected = False
                self.error_msg = f"無法開啟鏡頭 #{self.camera_num}: {str(e)}"
                return False, self.error_msg
        else:
            self.cap = cv2.VideoCapture(self.camera_num)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.is_connected = True
                self.error_msg = ""
                return True, f"成功啟動 USB 鏡頭 #{self.camera_num}"
            else:
                self.is_connected = False
                self.error_msg = f"找不到鏡頭編號 #{self.camera_num}"
                return False, self.error_msg

    def get_frame(self) -> np.ndarray:
        """從當前鏡頭擷取單張影像並完成方向與色彩修正。
        若鏡頭未連接，則回傳帶有錯誤提示的深色測試畫面，確保伺服器不崩潰。

        Args:
            無。

        Returns:
            np.ndarray: 修正後的彩色影像陣列，Shape 為 (H, W, 3)，dtype 為 uint8。
        """
        if self.is_connected:
            try:
                if HAS_PICAMERA2 and self.picam2 is not None:
                    frame = self.picam2.capture_array()
                elif self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        raise RuntimeError("無法讀取視訊幀")
                else:
                    raise RuntimeError("相機實例不存在")

                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                if self.swap_rb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            except Exception as e:
                self.is_connected = False
                self.error_msg = str(e)

        # 若無相機訊號，產生離線提示畫面 (仍可進行資料夾讀取與離線標定)
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            f"Camera #{self.camera_num} Offline / Disconnected",
            (40, self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 120, 255),
            2,
        )
        cv2.putText(
            placeholder,
            "You can still load existing folders & inspect calibration data.",
            (40, self.height // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            1,
        )
        return placeholder

    def release(self) -> None:
        """安全關閉並釋放目前佔用的相機硬體資源。

        Args:
            無。

        Returns:
            None
        """
        if HAS_PICAMERA2 and self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
            time.sleep(0.2)  # 給予 libcamera 底層釋放 pipeline 的緩衝時間
        elif self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False
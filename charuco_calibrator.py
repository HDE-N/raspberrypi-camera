import glob
import json
import os
import time
import cv2
import numpy as np

ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class CharucoCalibrator:
    """負責 ChArUco 標定板偵測、內外參矩陣解算，以及鏡頭專屬資料夾之影像與參數存取。"""

    def __init__(self):
        # 感光元件物理尺寸 (預設 Pi Camera v3 IMX708: 6.45 x 3.63 mm)
        self.sensor_width_mm = 6.45
        self.sensor_height_mm = 3.63

        # ChArUco 標定板預設規格
        self.dict_name = "DICT_5X5_100"
        self.squares_x = 7
        self.squares_y = 5
        self.square_length = 30.0  # 單位: mm
        self.marker_length = 22.0  # 單位: mm

        self.board = None
        self.detector = None
        self.rebuild_board(
            self.dict_name,
            self.squares_x,
            self.squares_y,
            self.square_length,
            self.marker_length,
        )

        # 樣本與標定結果狀態
        self.samples = []
        self.is_calibrated = False
        self.rms_error = 0.0
        self.camera_matrix = None  # Shape: (3, 3)
        self.dist_coeffs = None    # Shape: (1, 5)
        self.physical_metrics = {}
        self.calibrated_resolution = [1920, 1080]

    def rebuild_board(
        self,
        dict_name: str,
        squares_x: int,
        squares_y: int,
        square_length: float,
        marker_length: float,
    ) -> None:
        """重建 ChArUco 標定板與偵測器實例。

        Args:
            dict_name (str): ArUco 字典名稱 (如 'DICT_5X5_100')。
            squares_x (int): 水平方格數 (>= 3)。
            squares_y (int): 垂直方格數 (>= 3)。
            square_length (float): 棋盤方格邊長 (單位: mm)。
            marker_length (float): ArUco 標記邊長 (單位: mm，需小於 square_length)。

        Returns:
            None
        """
        self.dict_name = dict_name
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_length = float(square_length)
        self.marker_length = float(marker_length)

        dict_id = ARUCO_DICT_MAP.get(dict_name, cv2.aruco.DICT_5X5_100)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            aruco_dict,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect_and_draw(
        self, raw_frame: np.ndarray, draw_axes: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict | None]:
        """在輸入影像副本上偵測並繪製 ChArUco 角點，若已標定則同步計算即時外參並繪製 3D 座標軸。
        注意：不會修改傳入的 raw_frame，以確保儲存照片時為乾淨的原圖。

        Args:
            raw_frame (np.ndarray): 原始 BGR 彩色影像，Shape 為 (H, W, 3)。
            draw_axes (bool): 若已標定，是否繪製 3D 座標軸。

        Returns:
            tuple 包含:
                - annotated_frame (np.ndarray): 繪製標記與座標軸後的影像，Shape 為 (H, W, 3)。
                - charuco_corners (np.ndarray | None): 2D 亞像素角點，Shape 為 (N, 1, 2)。
                - charuco_ids (np.ndarray | None): 角點 ID，Shape 為 (N, 1)。
                - live_extrinsics (dict | None): 即時外參字典，未標定或角點不足時為 None。
        """
        annotated = raw_frame.copy()
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.detector.detectBoard(gray)
        )

        if marker_corners is not None and len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                annotated, charuco_corners, charuco_ids, (0, 255, 128)
            )

        live_extrinsics = None
        if (
            self.is_calibrated
            and charuco_ids is not None
            and len(charuco_ids) >= 4
        ):
            obj_pts, img_pts = self.match_object_and_image_points(
                charuco_corners, charuco_ids
            )
            if self.is_non_collinear(obj_pts):
                live_extrinsics, rvec, tvec = self.estimate_pose_pnp(
                    obj_pts, img_pts, self.camera_matrix, self.dist_coeffs
                )
                if live_extrinsics is not None and draw_axes:
                    cv2.drawFrameAxes(
                        annotated,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        self.square_length * 2.5,
                        5,
                    )

        return annotated, charuco_corners, charuco_ids, live_extrinsics

    def match_object_and_image_points(
        self, charuco_corners: np.ndarray, charuco_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """將 2D 影像角點與標定板上的 3D 物理空間點 (Z=0) 進行索引配對。

        Args:
            charuco_corners (np.ndarray): 2D 影像角點座標，Shape 為 (N, 1, 2)，單位 pixels。
            charuco_ids (np.ndarray): 對應的角點 ID 編號，Shape 為 (N, 1)。

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - obj_pts: 3D 標定板座標點，Shape 為 (N, 3)，單位 mm。
                - img_pts: 2D 影像像素座標點，Shape 為 (N, 1, 2)，單位 pixels。
        """
        chessboard_pts = self.board.getChessboardCorners()
        obj_pts = chessboard_pts[charuco_ids.flatten()].astype(np.float32)
        img_pts = charuco_corners.astype(np.float32)
        return obj_pts, img_pts

    @staticmethod
    def is_non_collinear(obj_pts: np.ndarray) -> bool:
        """檢查 3D 點集在 X-Y 平面上是否非共線 (Matrix Rank >= 2)。

        Args:
            obj_pts (np.ndarray): 3D 空間座標點，Shape 為 (N, 3)。

        Returns:
            bool: 非共線回傳 True，共線或點數不足回傳 False。
        """
        if len(obj_pts) < 4:
            return False
        centered = obj_pts[:, :2] - obj_pts[:, :2].mean(axis=0)
        return int(np.linalg.matrix_rank(centered)) >= 2

    @staticmethod
    def convert_rvec_tvec_to_extrinsics(
        rvec: np.ndarray, tvec: np.ndarray
    ) -> dict:
        """【外參核心轉換】將旋轉向量 rvec 與平移向量 tvec 轉換為旋轉矩陣 R、4x4 齊次轉換矩陣 T 與歐拉角。

        Args:
            rvec (np.ndarray): Rodrigues 旋轉向量，Shape 為 (3, 1)，單位 rad。
            tvec (np.ndarray): 平移向量，Shape 為 (3, 1)，單位 mm。

        Returns:
            dict: 包含 'R' (3x3), 't' (3,), 'rvec' (3,), 'T_4x4' (4x4), 'euler_deg' (roll/pitch/yaw) 與 'distance_mm'。
        """
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        T_4x4 = np.eye(4, dtype=np.float64)
        T_4x4[:3, :3] = R
        T_4x4[:3, 3] = t

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy >= 1e-6:
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = 0.0

        return {
            "R": R.tolist(),
            "t": t.tolist(),
            "rvec": rvec.flatten().tolist(),
            "T_4x4": T_4x4.tolist(),
            "euler_deg": {
                "roll": round(float(roll), 2),
                "pitch": round(float(pitch), 2),
                "yaw": round(float(yaw), 2),
            },
            "distance_mm": round(float(np.linalg.norm(t)), 2),
        }

    def estimate_pose_pnp(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[dict | None, np.ndarray | None, np.ndarray | None]:
        """【單張外參求解 PnP】已知內參 K 與畸變 D，由單張 3D-2D 點對求解相機外參。

        Args:
            obj_pts (np.ndarray): 3D 物理點，Shape 為 (N, 3)，單位 mm。
            img_pts (np.ndarray): 2D 像素點，Shape 為 (N, 1, 2)，單位 pixels。
            camera_matrix (np.ndarray): 內參矩陣 K，Shape 為 (3, 3)。
            dist_coeffs (np.ndarray): 畸變係數 D，Shape 為 (1, 5)。

        Returns:
            tuple: (extrinsics 字典, rvec 旋轉向量, tvec 平移向量)。
        """
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, dist_coeffs
        )
        if not ok:
            return None, None, None
        return self.convert_rvec_tvec_to_extrinsics(rvec, tvec), rvec, tvec

    def save_sample_to_disk(
        self,
        cam_dir: str,
        raw_frame: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
    ) -> dict:
        """將擷取到的原始影像儲存至 `<cam_dir>/images/sample_XXX.png`，並將配對點加入樣本池。

        Args:
            cam_dir (str): 該顆鏡頭對應的專屬資料夾路徑。
            raw_frame (np.ndarray): 未繪製標記的乾淨原始影像，Shape 為 (H, W, 3)。
            corners (np.ndarray): 偵測到的 2D 角點，Shape 為 (N, 1, 2)。
            ids (np.ndarray): 對應的角點 ID，Shape 為 (N, 1)。

        Returns:
            dict: 新建立的樣本摘要資訊 (含 id, filename, num_corners, error)。
        """
        img_dir = os.path.join(cam_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        sample_id = len(self.samples) + 1
        filename = f"sample_{sample_id:03d}.png"
        filepath = os.path.join(img_dir, filename)

        # 儲存無損 PNG 原始影像
        cv2.imwrite(filepath, raw_frame)

        obj_pts, img_pts = self.match_object_and_image_points(corners, ids)
        sample_record = {
            "id": sample_id,
            "filename": filename,
            "filepath": filepath,
            "obj_pts": obj_pts,
            "img_pts": img_pts,
            "num_corners": int(len(ids)),
            "error": None,
            "extrinsics": None,
        }
        self.samples.append(sample_record)

        return {
            "id": sample_id,
            "filename": filename,
            "num_corners": int(len(ids)),
            "error": None,
        }

    def calibrate_and_save(
        self, cam_dir: str, image_size: tuple[int, int]
    ) -> dict:
        """【內外參聯合標定並寫入磁碟】
        計算內參 K、畸變 D、光學指標與各樣本外參，並自動將結果存入 `cam_dir` 下的
        `calibration.json`、`calibration.yaml` 與 `calibration.npz`。

        Args:
            cam_dir (str): 該顆鏡頭對應的專屬資料夾路徑。
            image_size (tuple[int, int]): 影像解析度 (width, height)，單位 pixels。

        Returns:
            dict: 包含 rms_error, K, D, physical, samples 與儲存路徑的完整標定結果字典。
        """
        obj_points = [s["obj_pts"] for s in self.samples]
        img_points = [s["img_pts"] for s in self.samples]

        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )

        self.is_calibrated = True
        self.rms_error = round(float(rms), 4)
        self.camera_matrix = K
        self.dist_coeffs = D
        self.calibrated_resolution = [int(image_size[0]), int(image_size[1])]

        fovx, fovy, focal_mm, pp_mm, _ = cv2.calibrationMatrixValues(
            K, image_size, self.sensor_width_mm, self.sensor_height_mm
        )
        self.physical_metrics = {
            "fov_x_deg": round(float(fovx), 2),
            "fov_y_deg": round(float(fovy), 2),
            "focal_length_mm": round(float(focal_mm), 3),
            "principal_point_mm": [
                round(float(pp_mm[0]), 3),
                round(float(pp_mm[1]), 3),
            ],
            "sensor_size_mm": [self.sensor_width_mm, self.sensor_height_mm],
        }

        samples_output = []
        for idx, s in enumerate(self.samples):
            proj_pts, _ = cv2.projectPoints(
                s["obj_pts"], rvecs[idx], tvecs[idx], K, D
            )
            err = cv2.norm(s["img_pts"], proj_pts, cv2.NORM_L2) / np.sqrt(
                len(proj_pts)
            )
            s["error"] = round(float(err), 4)
            s["extrinsics"] = self.convert_rvec_tvec_to_extrinsics(
                rvecs[idx], tvecs[idx]
            )
            samples_output.append(
                {
                    "id": s["id"],
                    "filename": s["filename"],
                    "num_corners": s["num_corners"],
                    "error": s["error"],
                    "extrinsics": s["extrinsics"],
                }
            )

        # 自動將結果寫入該鏡頭的專屬資料夾
        os.makedirs(cam_dir, exist_ok=True)
        self._write_calibration_files(cam_dir, samples_output)

        return {
            "rms_error": self.rms_error,
            "K": K.tolist(),
            "D": D.flatten().tolist(),
            "physical": self.physical_metrics,
            "samples": samples_output,
            "saved_dir": os.path.abspath(cam_dir),
        }

    def _write_calibration_files(
        self, cam_dir: str, samples_output: list[dict]
    ) -> None:
        """將標定結果分別存為 JSON、YAML 與 NPZ 檔案至指定鏡頭資料夾。

        Args:
            cam_dir (str): 目標鏡頭資料夾路徑。
            samples_output (list[dict]): 包含每張樣本外參與誤差的列表。

        Returns:
            None
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 1. 儲存完整 JSON (供程式重新載入與跨平台讀取)
        json_payload = {
            "calibration_time": timestamp,
            "resolution": self.calibrated_resolution,
            "board_config": {
                "dict_name": self.dict_name,
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length": self.square_length,
                "marker_length": self.marker_length,
                "sensor_width_mm": self.sensor_width_mm,
                "sensor_height_mm": self.sensor_height_mm,
            },
            "rms_error": self.rms_error,
            "K": self.camera_matrix.tolist(),
            "D": self.dist_coeffs.flatten().tolist(),
            "physical": self.physical_metrics,
            "samples": samples_output,
        }
        with open(
            os.path.join(cam_dir, "calibration.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        # 2. 儲存 OpenCV 標準 YAML
        yaml_path = os.path.join(cam_dir, "calibration.yaml")
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
        fs.write("calibration_time", timestamp)
        fs.write("image_width", self.calibrated_resolution[0])
        fs.write("image_height", self.calibrated_resolution[1])
        fs.write("rms_error", self.rms_error)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("distortion_coefficients", self.dist_coeffs)
        fs.release()

        # 3. 儲存 NumPy NPZ
        np.savez(
            os.path.join(cam_dir, "calibration.npz"),
            K=self.camera_matrix,
            D=self.dist_coeffs,
            rms=self.rms_error,
            resolution=np.array(self.calibrated_resolution),
            R_list=np.array([s["extrinsics"]["R"] for s in samples_output]),
            t_list=np.array([s["extrinsics"]["t"] for s in samples_output]),
            T_4x4_list=np.array(
                [s["extrinsics"]["T_4x4"] for s in samples_output]
            ),
        )

    def load_from_directory(self, cam_dir: str) -> dict:
        """【讀取鏡頭資料夾】
        打開指定的鏡頭資料夾，自動讀取：
        1. `calibration.json` 內的標定板設定、相機內參 K、畸變係數 D、光學指標與歷史外參。
        2. `images/` 資料夾內的所有標定照片，重新提取角點以支援隨時增刪樣本與重新計算。

        Args:
            cam_dir (str): 要載入的鏡頭資料夾路徑。

        Returns:
            dict: 載入後的完整狀態字典 (含是否有歷史標定檔、載入照片數量、K、D、samples 列表與標定板設定)。
        """
        os.makedirs(os.path.join(cam_dir, "images"), exist_ok=True)
        self.samples.clear()
        self.is_calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None

        json_path = os.path.join(cam_dir, "calibration.json")
        saved_samples_map = {}

        # Step 1: 若存在 calibration.json，先還原標定板參數與內外參矩陣
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            b_cfg = data.get("board_config", {})
            if b_cfg:
                self.sensor_width_mm = float(
                    b_cfg.get("sensor_width_mm", self.sensor_width_mm)
                )
                self.sensor_height_mm = float(
                    b_cfg.get("sensor_height_mm", self.sensor_height_mm)
                )
                self.rebuild_board(
                    b_cfg.get("dict_name", self.dict_name),
                    b_cfg.get("squares_x", self.squares_x),
                    b_cfg.get("squares_y", self.squares_y),
                    b_cfg.get("square_length", self.square_length),
                    b_cfg.get("marker_length", self.marker_length),
                )

            if "K" in data and "D" in data:
                self.is_calibrated = True
                self.rms_error = float(data.get("rms_error", 0.0))
                self.camera_matrix = np.array(data["K"], dtype=np.float64)
                self.dist_coeffs = np.array(data["D"], dtype=np.float64).reshape(1, -1)
                self.physical_metrics = data.get("physical", {})
                self.calibrated_resolution = data.get("resolution", [1920, 1080])

            for s in data.get("samples", []):
                if "filename" in s:
                    saved_samples_map[s["filename"]] = s

        # Step 2: 掃描 images/ 目錄下的所有影像並偵測角點
        img_files = sorted(
            glob.glob(os.path.join(cam_dir, "images", "*.png"))
            + glob.glob(os.path.join(cam_dir, "images", "*.jpg"))
        )

        samples_summary = []
        for filepath in img_files:
            filename = os.path.basename(filepath)
            img = cv2.imread(filepath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = self.detector.detectBoard(gray)
            if ids is None or len(ids) < 4:
                continue

            obj_pts, img_pts = self.match_object_and_image_points(corners, ids)
            if not self.is_non_collinear(obj_pts):
                continue

            sample_id = len(self.samples) + 1
            saved_info = saved_samples_map.get(filename, {})
            err = saved_info.get("error", None)
            ext = saved_info.get("extrinsics", None)

            # 若已有內參但該照片尚無外參，即時用 PnP 補算外參
            if self.is_calibrated and ext is None:
                ext, _, _ = self.estimate_pose_pnp(
                    obj_pts, img_pts, self.camera_matrix, self.dist_coeffs
                )

            record = {
                "id": sample_id,
                "filename": filename,
                "filepath": filepath,
                "obj_pts": obj_pts,
                "img_pts": img_pts,
                "num_corners": int(len(ids)),
                "error": err,
                "extrinsics": ext,
            }
            self.samples.append(record)
            samples_summary.append(
                {
                    "id": sample_id,
                    "filename": filename,
                    "num_corners": int(len(ids)),
                    "error": err,
                    "extrinsics": ext,
                }
            )

        return {
            "is_calibrated": self.is_calibrated,
            "rms_error": self.rms_error if self.is_calibrated else None,
            "K": self.camera_matrix.tolist() if self.is_calibrated else None,
            "D": self.dist_coeffs.flatten().tolist() if self.is_calibrated else None,
            "physical": self.physical_metrics if self.is_calibrated else None,
            "samples": samples_summary,
            "board_config": {
                "dict_name": self.dict_name,
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length": self.square_length,
                "marker_length": self.marker_length,
                "sensor_width_mm": self.sensor_width_mm,
                "sensor_height_mm": self.sensor_height_mm,
            },
        }
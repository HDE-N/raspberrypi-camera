import io
import os
import shutil
import threading
import time
import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file

from camera_manager import CameraManager
from charuco_calibrator import CharucoCalibrator

app = Flask(__name__)
state_lock = threading.Lock()

# 初始化相機與標定器 (預設使用 1920x1080 與鏡頭 0)
camera = CameraManager(camera_num=0, width=1920, height=1080, rotate_180=True, swap_rb=True)
calibrator = CharucoCalibrator()

# 工作區與鏡頭資料夾狀態
workspace_state = {
    "base_dir": os.path.abspath("./calib_workspace"),
    "camera_id": "cam_0",  # 對應子資料夾 ./calib_workspace/cam_0
}

# 即時串流共享狀態
live_state = {
    "fps": 0.0,
    "corner_count": 0,
    "show_axes": True,
    "latest_raw_frame": None,
    "latest_corners": None,
    "latest_ids": None,
    "live_extrinsics": None,
}


def get_current_cam_dir() -> str:
    """取得目前鏡頭對應的完整資料夾路徑 `<base_dir>/<camera_id>`。

    Returns:
        str: 目前鏡頭的資料夾絕對路徑。
    """
    cam_dir = os.path.join(workspace_state["base_dir"], workspace_state["camera_id"])
    os.makedirs(os.path.join(cam_dir, "images"), exist_ok=True)
    return cam_dir


# 啟動時先自動載入預設鏡頭資料夾
calibrator.load_from_directory(get_current_cam_dir())


def generate_mjpeg_stream():
    """產生即時 MJPEG 影像串流，並保存最新一幀乾淨原圖供拍照存檔使用。"""
    prev_time = time.time()
    while True:
        with state_lock:
            raw_frame = camera.get_frame()
            annotated_frame, corners, ids, live_ext = calibrator.detect_and_draw(
                raw_frame, draw_axes=live_state["show_axes"]
            )

            live_state["latest_raw_frame"] = raw_frame
            live_state["latest_corners"] = corners
            live_state["latest_ids"] = ids
            live_state["corner_count"] = len(ids) if ids is not None else 0
            live_state["live_extrinsics"] = live_ext

            curr_time = time.time()
            live_state["fps"] = round(1.0 / max(curr_time - prev_time, 1e-4), 1)
            prev_time = curr_time

            ret, buffer = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )

        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        time.sleep(0.01)


@app.route("/")
def index():
    """渲染前端主頁面。"""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """提供即時串流影像。"""
    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """回傳即時 FPS、角點數、相機連線狀態與即時外參。"""
    with state_lock:
        max_corners = (calibrator.squares_x - 1) * (calibrator.squares_y - 1)
        return jsonify(
            {
                "fps": live_state["fps"],
                "corner_count": live_state["corner_count"],
                "max_corners": max_corners,
                "is_calibrated": calibrator.is_calibrated,
                "camera_connected": camera.is_connected,
                "camera_num": camera.camera_num,
                "camera_id": workspace_state["camera_id"],
                "cam_dir": get_current_cam_dir(),
                "live_extrinsics": live_state["live_extrinsics"],
            }
        )


@app.route("/api/list_workspace", methods=["GET"])
def api_list_workspace():
    """掃描工作區根目錄下的所有鏡頭子資料夾，回傳各資料夾的照片數與標定狀態。"""
    with state_lock:
        base_dir = workspace_state["base_dir"]
        os.makedirs(base_dir, exist_ok=True)
        folders = []
        for entry in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, entry)
            if os.path.isdir(full_path):
                img_dir = os.path.join(full_path, "images")
                img_count = 0
                if os.path.isdir(img_dir):
                    img_count = len(
                        [
                            f
                            for f in os.listdir(img_dir)
                            if f.endswith((".png", ".jpg"))
                        ]
                    )
                has_calib = os.path.exists(
                    os.path.join(full_path, "calibration.json")
                )
                folders.append(
                    {
                        "camera_id": entry,
                        "image_count": img_count,
                        "is_calibrated": has_calib,
                    }
                )
        return jsonify(
            {
                "base_dir": base_dir,
                "current_camera_id": workspace_state["camera_id"],
                "current_camera_num": camera.camera_num,
                "folders": folders,
                "hw_cameras": camera.list_available_cameras(),
            }
        )


@app.route("/api/switch_or_load", methods=["POST"])
def api_switch_or_load():
    """切換工作區路徑、鏡頭編號與對應資料夾，並載入該資料夾內的既有照片與內外參數據。"""
    data = request.json
    with state_lock:
        base_dir = data.get("base_dir", workspace_state["base_dir"]).strip()
        camera_id = data.get("camera_id", workspace_state["camera_id"]).strip()
        new_cam_num = int(data.get("camera_num", camera.camera_num))

        if not camera_id:
            return jsonify({"ok": False, "msg": "鏡頭資料夾名稱不可為空"}), 400

        workspace_state["base_dir"] = os.path.abspath(base_dir)
        workspace_state["camera_id"] = camera_id
        cam_dir = get_current_cam_dir()

        # 若硬體鏡頭編號改變，切換硬體鏡頭
        hw_msg = ""
        if new_cam_num != camera.camera_num or not camera.is_connected:
            ok, msg = camera.init_hardware(new_cam_num, camera.width, camera.height)
            hw_msg = f" | {msg}"

        # 載入該資料夾的歷史照片與標定參數
        loaded_data = calibrator.load_from_directory(cam_dir)
        return jsonify(
            {
                "ok": True,
                "msg": f"已載入資料夾 [{cam_dir}]：共 {len(loaded_data['samples'])} 張照片"
                + (" (含已標定內外參)" if loaded_data["is_calibrated"] else " (尚未標定)")
                + hw_msg,
                "cam_dir": cam_dir,
                **loaded_data,
            }
        )


@app.route("/api/update_config", methods=["POST"])
def api_update_config():
    """更新相機解析度、畫面方向與 ChArUco 標定板參數。"""
    data = request.json
    with state_lock:
        new_w = int(data.get("width", camera.width))
        new_h = int(data.get("height", camera.height))
        new_cam_num = int(data.get("camera_num", camera.camera_num))
        sq_x = int(data.get("squares_x", calibrator.squares_x))
        sq_y = int(data.get("squares_y", calibrator.squares_y))
        sq_len = float(data.get("square_length", calibrator.square_length))
        mk_len = float(data.get("marker_length", calibrator.marker_length))
        dict_name = data.get("dict_name", calibrator.dict_name)

        if sq_x < 3 or sq_y < 3:
            return jsonify({"ok": False, "msg": "方格行列數必須 >= 3"}), 400
        if mk_len >= sq_len or mk_len <= 0:
            return jsonify({"ok": False, "msg": "標記邊長必須小於方格邊長"}), 400

        camera.rotate_180 = bool(data.get("rotate_180", camera.rotate_180))
        camera.swap_rb = bool(data.get("swap_rb", camera.swap_rb))
        live_state["show_axes"] = bool(data.get("show_axes", live_state["show_axes"]))

        if (
            new_w != camera.width
            or new_h != camera.height
            or new_cam_num != camera.camera_num
        ):
            camera.init_hardware(new_cam_num, new_w, new_h)

        calibrator.sensor_width_mm = float(
            data.get("sensor_width_mm", calibrator.sensor_width_mm)
        )
        calibrator.sensor_height_mm = float(
            data.get("sensor_height_mm", calibrator.sensor_height_mm)
        )
        calibrator.rebuild_board(dict_name, sq_x, sq_y, sq_len, mk_len)

        return jsonify({"ok": True, "msg": "相機與標定板設定已更新！"})


@app.route("/api/capture", methods=["POST"])
def api_capture():
    """擷取當前畫面，自動將照片存檔至 `<cam_dir>/images/` 並加入樣本池。"""
    with state_lock:
        raw_frame = live_state["latest_raw_frame"]
        corners = live_state["latest_corners"]
        ids = live_state["latest_ids"]

        if raw_frame is None or ids is None or len(ids) < 6:
            return jsonify({"ok": False, "msg": "角點不足 6 點，無法擷取儲存！"}), 400

        obj_pts, _ = calibrator.match_object_and_image_points(corners, ids)
        if not calibrator.is_non_collinear(obj_pts):
            return jsonify({"ok": False, "msg": "偵測角點共線，請調整標定板角度！"}), 400

        cam_dir = get_current_cam_dir()
        new_sample = calibrator.save_sample_to_disk(
            cam_dir, raw_frame, corners, ids
        )

        summary = [
            {
                "id": s["id"],
                "filename": s["filename"],
                "num_corners": s["num_corners"],
                "error": s["error"],
                "extrinsics": s["extrinsics"],
            }
            for s in calibrator.samples
        ]
        return jsonify(
            {
                "ok": True,
                "msg": f"已儲存照片 {new_sample['filename']} 至 {cam_dir}/images/",
                "samples": summary,
            }
        )


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """計算內外參並自動將 JSON/YAML/NPZ 檔案寫入目前鏡頭資料夾。"""
    with state_lock:
        if len(calibrator.samples) < 4:
            return jsonify({"ok": False, "msg": "樣本數不足 4 張，請先拍攝或載入更多照片！"}), 400

        cam_dir = get_current_cam_dir()
        result = calibrator.calibrate_and_save(
            cam_dir, (camera.width, camera.height)
        )
        return jsonify(
            {
                "ok": True,
                "msg": f"標定完成！結果已自動儲存至 {result['saved_dir']}",
                **result,
            }
        )


@app.route("/api/clear_samples", methods=["POST"])
def api_clear_samples():
    """清空目前鏡頭資料夾內的圖片與標定檔。"""
    with state_lock:
        cam_dir = get_current_cam_dir()
        if os.path.exists(cam_dir):
            shutil.rmtree(cam_dir)
        os.makedirs(os.path.join(cam_dir, "images"), exist_ok=True)
        calibrator.samples.clear()
        calibrator.is_calibrated = False
        return jsonify({"ok": True, "msg": f"已清空資料夾 {cam_dir} 的所有照片與標定數據"})


@app.route("/api/sample_image/<filename>")
def api_sample_image(filename: str):
    """讀取並回傳指定樣本的已儲存影像（加上繪製後的角點方便檢視）。"""
    with state_lock:
        filepath = os.path.join(get_current_cam_dir(), "images", filename)
        if not os.path.exists(filepath):
            return "File not found", 404
        img = cv2.imread(filepath)
        annotated, _, _, _ = calibrator.detect_and_draw(img, draw_axes=True)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/board_image")
def api_board_image():
    """下載目前設定的 ChArUco 標定板 PNG。"""
    with state_lock:
        img = calibrator.board.generateImage(
            (calibrator.squares_x * 200, calibrator.squares_y * 200),
            marginSize=40,
        )
        _, buf = cv2.imencode(".png", img)
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/png",
            as_attachment=True,
            download_name=f"ChArUco_{calibrator.dict_name}.png",
        )


@app.route("/api/export/<fmt>")
def api_export(fmt: str):
    """從目前鏡頭資料夾下載已儲存的標定檔案 (json / yaml / npz)。"""
    with state_lock:
        cam_dir = get_current_cam_dir()
        ext_map = {
            "json": ("calibration.json", "application/json"),
            "yaml": ("calibration.yaml", "text/yaml"),
            "npz": ("calibration.npz", "application/octet-stream"),
        }
        if fmt not in ext_map:
            return jsonify({"ok": False, "msg": "不支援的格式"}), 400

        fname, mime = ext_map[fmt]
        fpath = os.path.join(cam_dir, fname)
        if not os.path.exists(fpath):
            return jsonify({"ok": False, "msg": "尚未產生標定檔案，請先執行計算內外參"}), 400

        download_name = f"{workspace_state['camera_id']}_{fname}"
        return send_file(fpath, mimetype=mime, as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    try:
        print("啟動 ChArUco 多鏡頭標定工作站：http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    finally:
        camera.release()
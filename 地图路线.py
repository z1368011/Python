# map.py  —  车辆轨迹可视化后端（Flask + Folium）
# 启动：python map.py --host 0.0.0.0 --port 5000

import glob
import json
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import List
# 在导入部分添加
from folium.plugins import TimestampedGeoJson
import folium
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from geopy.distance import geodesic

# ────────────────────────── 全局配置 ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "data_path": os.path.join(BASE_DIR, "taxi_data_split"),  # 直接指向taxi_data_split
    "json_path": os.path.join(BASE_DIR, "深圳市.json"),  # 直接指向深圳市.json
    "output_folder": os.path.join(BASE_DIR, "map_output"),  # 改为英文路径避免编码问题
    "static_url": "/static",  # 静态文件URL前缀
    "max_vehicles": 10000,
    "time_range": 60,
    "map_center": [22.52847, 114.05454],
    "default_zoom": 12,
    # 新增：按分钟快照文件夹配置
    "snapshot_folder": r"taxi_by_minute",
    "snapshot_pattern": "%Y%m%d_%H%M",
    "snapshot_ext": ".csv",
}

# ────────────────────────── 目录初始化 ─────────────────────────
app = Flask(__name__, static_url_path='', static_folder='map_output/static')

output_folder = CONFIG["output_folder"]
static_dir = os.path.join(output_folder, "static")
resources_dir = os.path.join(BASE_DIR, "resources")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(resources_dir, exist_ok=True)


# ────────────────────────── 工具函数 ───────────────────────────
def load_shenzhen_boundary():
    try:
        with open(CONFIG["json_path"], encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Boundary] 读取失败：{e}")
        return None


def get_available_vehicles():
    try:
        return [
            f.replace(".csv", "")
            for f in os.listdir(CONFIG["data_path"])
            if f.endswith(".csv")
        ]
    except FileNotFoundError:
        return []


def load_vehicle_data(vehicle_id, start_time=None, end_time=None):
    fp = os.path.join(CONFIG["data_path"], f"{vehicle_id}.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp, parse_dates=['time'])

    # 列名标准化
    df.rename(columns={
        'lat': 'latitude', 'lon': 'longitude',
        'lati': 'latitude', 'long': 'longitude'}, inplace=True)

    if start_time and end_time:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        df = df.loc[mask]

    if 'id' not in df.columns:
        df['id'] = vehicle_id
    if 'speed' not in df.columns:
        df['speed'] = np.nan

    # 计算相邻点距离
    if not df.empty:
        df['distance_to_next'] = df.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                (df.iloc[row.name + 1]['latitude'], df.iloc[row.name + 1]['longitude'])
            ).km if row.name < len(df) - 1 else 0,
            axis=1
        )
    return df


# ──────────── 时间点快照地图（基于分钟文件夹） ──────────────
def generate_snapshot_map(time_point: datetime, vehicle_ids: List[str] = None, tolerance_minutes=5):
    fname = time_point.strftime(CONFIG["snapshot_pattern"]) + CONFIG["snapshot_ext"]
    fpath = os.path.join(CONFIG["snapshot_folder"], fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"快照文件不存在: {fpath}")

    # 修改后的CSV读取方式 - 增加容错处理
    try:
        df = pd.read_csv(fpath, header=None, skiprows=1,
                         names=["id", "time", "lat", "lon", "status", "speed"],
                         on_bad_lines='skip')  # 关键修改：跳过错误行
    except Exception as e:
        print(f"[ERROR] CSV读取失败: {str(e)}")
        raise

    print(f"[DEBUG] 读取文件 {fpath}，共{len(df)}行数据")

    # 筛选车辆
    if vehicle_ids:
        df = df[df["id"].astype(str).isin(vehicle_ids)]

    print(f"[DEBUG] 筛选车辆后剩余{len(df)}行")

    if df.empty:
        raise ValueError("该时间点无指定车辆数据")

    # 经纬度类型转换
    # 注意：文件里列“lat”其实是经度，“lon”其实是纬度，folium要求 [纬度, 经度]
    df['latitude'] = pd.to_numeric(df['lon'], errors='coerce')  # 纬度
    df['longitude'] = pd.to_numeric(df['lat'], errors='coerce')  # 经度

    # 丢弃无效经纬度
    df = df.dropna(subset=['latitude', 'longitude'])

    # 过滤合理范围
    valid_df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]

    print(f"[DEBUG] 经纬度有效数据: {len(valid_df)}")

    m = folium.Map(location=CONFIG["map_center"], zoom_start=CONFIG["default_zoom"])

    # 加载边界
    boundary = load_shenzhen_boundary()
    if boundary:
        folium.GeoJson(boundary, name="深圳市边界").add_to(m)

    colors = [
        "red", "blue", "green", "purple", "orange", "darkred",
        "lightred", "beige", "darkblue", "darkgreen", "cadetblue",
        "darkpurple", "white", "pink", "lightblue", "lightgreen",
        "gray", "black", "lightgray"
    ]

    plotted = 0
    for idx, (_, row) in enumerate(valid_df.iterrows()):
        lat = row['latitude']
        lon = row['longitude']

        folium.CircleMarker(
            location=[lat, lon],  # 纬度，经度顺序
            radius=6,
            color=colors[idx % len(colors)],
            fill=True,
            fill_opacity=0.8,
            popup=(f"车辆: {row['id']}<br>"
                   f"时间: {row['time']}<br>"
                   f"速度: {row['speed']} km/h<br>"
                   f"状态: {row['status']}")
        ).add_to(m)
        plotted += 1

    out_name = f"snapshot_{int(time.time())}.html"
    out_path = os.path.join(static_dir, out_name)
    m.save(out_path)
    print(f"[DEBUG] 地图保存为 {out_path}")

    return out_name, plotted


# ──────────── 动画轨迹地图（使用纯JavaScript实现） ───────────────
def generate_animated_map(vehicle_id: str, start_time: datetime, minutes_forward=30):
    """使用Folium TimestampedGeoJson实现动画轨迹"""
    end_time = start_time + timedelta(minutes=minutes_forward)
    df = load_vehicle_data(vehicle_id, start_time, end_time)

    # 详细的调试输出
    print(f"[动画轨迹] 车辆 {vehicle_id} 查询: {start_time} ~ {end_time}")
    print(f"数据点数: {len(df)}")

    if df.empty:
        print("错误: 查询结果为空")
        return None
    if len(df) < 2:
        print(f"警告: 只有 {len(df)} 个点，无法形成轨迹")
        return None

    # 确保时间列是 datetime 类型
    if not isinstance(df['time'].iloc[0], datetime):
        df['time'] = pd.to_datetime(df['time'])

    # 创建基础地图 - 使用轨迹的第一个点作为中心点
    init_lat = df.iloc[0]['latitude']
    init_lng = df.iloc[0]['longitude']

    print(f"地图中心点: [{init_lat}, {init_lng}]")

    m = folium.Map(
        location=[init_lat, init_lng],
        zoom_start=CONFIG["default_zoom"],
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )

    # 添加深圳市边界
    boundary = load_shenzhen_boundary()
    if boundary:
        folium.GeoJson(boundary, name="深圳市边界").add_to(m)

    # 准备GeoJSON格式的轨迹数据
    features = []

    # 1. 添加完整轨迹线
    line_coords = []
    for _, row in df.iterrows():
        line_coords.append([row['longitude'], row['latitude']])

    line_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": line_coords
        },
        "properties": {
            "times": [dt.isoformat() for dt in df['time']],
            "style": {
                "color": "#4169E1",
                "weight": 4,
                "opacity": 0.7
            },
            "icon": "circle",
            "iconstyle": {
                "fillColor": "blue",
                "fillOpacity": 0.8,
                "stroke": "true",
                "radius": 4
            }
        }
    }
    features.append(line_feature)

    # 2. 添加移动点（车辆位置）
    for i, (_, row) in enumerate(df.iterrows()):
        point_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            "properties": {
                "time": row['time'].isoformat(),
                "popup": f"时间: {row['time'].strftime('%H:%M:%S')}<br>速度: {row.get('speed', '-')} km/h",
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "#FF4500" if i != len(df) - 1 else "#32CD32",
                    "fillOpacity": 0.9,
                    "stroke": "true",
                    "radius": 7
                }
            }
        }
        features.append(point_feature)

    # 3. 添加起始点和终点标记
    start_point = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [df.iloc[0]['longitude'], df.iloc[0]['latitude']]
        },
        "properties": {
            "time": df['time'].min().isoformat(),
            "popup": f"起点<br>时间: {df.iloc[0]['time'].strftime('%H:%M:%S')}",
            "icon": "flag",
            "iconstyle": {
                "fillColor": "#32CD32",
                "fillOpacity": 1,
                "stroke": "true",
                "radius": 8
            }
        }
    }
    features.append(start_point)

    end_point = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [df.iloc[-1]['longitude'], df.iloc[-1]['latitude']]
        },
        "properties": {
            "time": df['time'].max().isoformat(),
            "popup": f"终点<br>时间: {df.iloc[-1]['time'].strftime('%H:%M:%S')}",
            "icon": "flag",
            "iconstyle": {
                "fillColor": "#FF0000",
                "fillOpacity": 1,
                "stroke": "true",
                "radius": 8
            }
        }
    }
    features.append(end_point)

    # 创建GeoJSON对象
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # 添加TimestamedGeoJson插件
    TimestampedGeoJson(
        geojson_data,
        period="PT1M",  # 每分钟播放一个点（可根据数据密度调整）
        add_last_point=True,
        auto_play=True,
        loop=False,
        max_speed=10,
        loop_button=True,
        date_options="YYYY/MM/DD HH:mm:ss",
        time_slider_drag_update=True,
        duration="PT2M",  # 动画总时长
    ).add_to(m)

    # 添加标题和信息
    title_html = f'''
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2);">
            <h4 style="margin:0">车辆 {vehicle_id} 轨迹动画</h4>
            <p style="margin:5px 0">时间段: {start_time.strftime('%Y-%m-%d %H:%M')} 至 {end_time.strftime('%Y-%m-%d %H:%M')}</p>
            <p style="margin:5px 0">数据点: {len(df)}个</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存地图
    fname = f"anim_{vehicle_id}_{int(time.time())}.html"
    output_path = os.path.join(static_dir, fname)
    m.save(output_path)
    print(f"动画地图保存至: {output_path}")
    return fname

# ──────────── 单车静态轨迹地图 ────────────────
# ──────────── 单车静态轨迹地图 ────────────────
def generate_vehicle_map(vehicle_id, df):
    if df.empty:
        return None

    # 创建地图
    m = folium.Map(location=CONFIG["map_center"], zoom_start=CONFIG["default_zoom"])

    # 添加边界
    boundary = load_shenzhen_boundary()
    if boundary:
        folium.GeoJson(boundary, name='深圳市边界').add_to(m)

    # 获取坐标列表
    coords = df[['latitude', 'longitude']].values.tolist()

    # 添加轨迹线上的点标记
    for idx, (_, row) in enumerate(df.iterrows()):
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color="green",
            popup=(f"时间:{row['time']}<br>"
                   f"速度:{row['speed']}km/h<br>"
                   f"距离:{row.get('distance_to_next', '-')}km")
        ).add_to(m)

    # 添加轨迹线
    folium.PolyLine(
        coords,
        color='blue',
        weight=2.5,
        popup=f"车辆 {vehicle_id} 轨迹"
    ).add_to(m)

    # 添加起点和终点标记（修正：传单个位置而非整个列表）
    if coords:
        # 起点标记
        folium.Marker(
            location=coords[0],
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            popup=f"起始点<br>时间:{df['time'].iloc[0]}"
        ).add_to(m)

        # 终点标记
        folium.Marker(
            location=coords[-1],
            icon=folium.Icon(color='red', icon='stop', prefix='fa'),
            popup=f"结束点<br>时间:{df['time'].iloc[-1]}"
        ).add_to(m)

    # 保存地图
    fname = f"vehicle_{vehicle_id}_{int(time.time())}.html"
    output_path = os.path.join(static_dir, fname)
    m.save(output_path)
    print(f"静态轨迹地图保存至: {output_path}")
    return fname

# ────────────────────────── 杂项：清理旧文件 ──────────────────
def cleanup_old_files(max_files=20):
    files = glob.glob(os.path.join(static_dir, "vehicle_*.html")) + \
            glob.glob(os.path.join(static_dir, "snapshot_*.html")) + \
            glob.glob(os.path.join(static_dir, "anim_*.html"))
    files.sort(key=os.path.getmtime)
    for f in files[:-max_files]:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[Cleanup] {f} 删除失败：{e}")


# ────────────────────────── Flask 路由 ────────────────────────
@app.route('/')
def index():
    vehicles = get_available_vehicles()[:CONFIG["max_vehicles"]]
    return render_template('index.html', vehicles=vehicles, config=CONFIG)


# —— 单车静态轨迹 ——————————————————————————
@app.route('/get_vehicle_data', methods=['POST'])
def get_vehicle_data():
    try:
        d = request.json
        vid = d['vehicle_id']
        st = datetime.fromisoformat(d['start_time']) if d['start_time'] else None
        et = datetime.fromisoformat(d['end_time']) if d['end_time'] else None
        df = load_vehicle_data(vid, st, et)
        if d.get('road_correction'): pass  # 预留
        fname = generate_vehicle_map(vid, df)
        cleanup_old_files()
        return jsonify({
            "success": True,
            "vehicle_id": vid,
            "point_count": len(df),
            "start_time": df['time'].min().isoformat() if not df.empty else "",
            "end_time": df['time'].max().isoformat() if not df.empty else "",
            "map_url": f"{CONFIG['static_url']}/{fname}" if fname else ""
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


# —— 时间点快照 ————————————————————————————
@app.route('/get_timepoint_snapshot', methods=['POST'])
def get_timepoint_snapshot():
    try:
        d = request.json
        tp = datetime.fromisoformat(d['time_point'])
        vids = d.get('vehicle_ids')  # 可为 None
        fname, count = generate_snapshot_map(tp, vids)
        cleanup_old_files()
        return jsonify({
            "success": True,
            "vehicle_count": count,
            "map_url": f"{CONFIG['static_url']}/{fname}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


# —— 动画轨迹 ——————————————————————————————
@app.route('/get_animated_trajectory', methods=['POST'])
def get_animated_trajectory():
    try:
        d = request.json
        vid = d['vehicle_id']
        st = datetime.fromisoformat(d['start_time'])
        print(f"请求动画轨迹: 车辆 {vid}, 起始时间 {st}")

        fname = generate_animated_map(vid, st)
        if not fname:
            return jsonify({"success": False, "error": "无有效数据点"}), 404

        cleanup_old_files()
        return jsonify({
            "success": True,
            "map_url": f"{CONFIG['static_url']}/{fname}",
            "debug_info": f"文件已生成: {fname}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


# —— 资源 / 静态文件直通 ———————————————————————
@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory(static_dir, filename)


@app.route('/resources/<path:filename>')
def resources_files(filename):
    return send_from_directory(resources_dir, filename)


# ────────────────────────── 启动 ─────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
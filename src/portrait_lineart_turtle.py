#!/usr/bin/env python3
"""
portrait_lineart_turtle.py

Portrait -> line-art, vẽ từng nét bằng turtle.

Chỉnh:
- Nét mảnh hơn.
- Lưu trực tiếp PNG từ paths bằng PIL (không phụ thuộc PostScript / Ghostscript).
"""
import argparse
import math
import time
from typing import List, Tuple, Optional
import os
import datetime

from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import turtle
import sys

# ---------- Default canvas and visuals ----------
CANVAS_W, CANVAS_H = 900, 900
BG_COLOR = "white"
LINE_COLOR = "black"
LINE_WIDTH = 0.6   # mặc định bút mảnh hơn

# ---------- Image loading ----------
def load_image_gray(path: str, max_size=(700,700)) -> Image.Image:
    img = Image.open(path).convert("L")
    img.thumbnail(max_size, Image.LANCZOS)
    return img

# ---------- Convolution fallback ----------
def convolve2d_np(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh//2, kw//2
    a_p = np.pad(a, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')
    h, w = a.shape
    out = np.zeros_like(a, dtype=float)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i,j] * a_p[i:i+h, j:j+w]
    return out

# ---------- Edge detection (Sobel) ----------
def sobel_edges(img: Image.Image, blur_radius: float = 1.0, edge_mul: float = 1.0) -> np.ndarray:
    """Return binary edge map (uint8 0/1)."""
    if blur_radius and blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    a = np.array(img, dtype=float)
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)
    try:
        from scipy.signal import convolve2d
        gx = convolve2d(a, Kx, mode='same', boundary='symm')
        gy = convolve2d(a, Ky, mode='same', boundary='symm')
    except Exception:
        gx = convolve2d_np(a, Kx)
        gy = convolve2d_np(a, Ky)
    g = np.hypot(gx, gy)
    if g.max() > 0:
        g = g / (g.max())
    p90 = np.percentile(g, 90)
    thr = max(0.06, p90 * 0.5 * edge_mul)
    edges = (g >= thr).astype(np.uint8)
    return edges

# ---------- Trace connected edge pixels into ordered paths ----------
def find_connected_paths(edges: np.ndarray) -> List[List[Tuple[int,int]]]:
    h, w = edges.shape
    visited = np.zeros_like(edges, dtype=bool)
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    paths: List[List[Tuple[int,int]]] = []
    for y in range(h):
        for x in range(w):
            if edges[y,x] and not visited[y,x]:
                cur = (x,y)
                visited[y,x] = True
                path = [cur]
                prev = None
                # walk forward greedily preferring straight continuation
                while True:
                    cx, cy = cur
                    neighs = []
                    for dx,dy in dirs:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h and edges[ny,nx] and not visited[ny,nx]:
                            neighs.append((nx,ny))
                    if not neighs:
                        break
                    if prev is None:
                        nxt = neighs[0]
                    else:
                        vx, vy = cx - prev[0], cy - prev[1]
                        best = None; best_ang = 10.0
                        for n in neighs:
                            ux, uy = n[0]-cx, n[1]-cy
                            dot = vx*ux + vy*uy
                            vlen = math.hypot(vx,vy) + 1e-9
                            ulen = math.hypot(ux,uy) + 1e-9
                            cosang = max(-1.0, min(1.0, dot/(vlen*ulen)))
                            ang = math.acos(cosang)
                            if ang < best_ang:
                                best_ang = ang; best = n
                        nxt = best if best is not None else neighs[0]
                    path.append(nxt)
                    visited[nxt[1], nxt[0]] = True
                    prev = cur
                    cur = nxt
                # backward extension from start
                start = path[0]
                cur = start
                back = []
                while True:
                    cx, cy = cur
                    neighs = []
                    for dx,dy in dirs:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h and edges[ny,nx] and not visited[ny,nx]:
                            neighs.append((nx,ny))
                    if not neighs:
                        break
                    nxt = neighs[0]
                    back.append(nxt)
                    visited[nxt[1], nxt[0]] = True
                    cur = nxt
                if back:
                    back.reverse()
                    path = back + path
                if len(path) >= 3:
                    paths.append(path)
    paths.sort(key=lambda p: -len(p))
    return paths

# ---------- Ramer-Douglas-Peucker ----------
def rdp(points: List[Tuple[int,int]], eps: float) -> List[Tuple[int,int]]:
    if len(points) < 3:
        return points[:]
    x1,y1 = points[0]; x2,y2 = points[-1]
    den = math.hypot(x2-x1, y2-y1) + 1e-9
    maxd = 0.0; idx = 0
    for i in range(1, len(points)-1):
        x0,y0 = points[i]
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        d = num/den
        if d > maxd:
            maxd = d; idx = i
    if maxd > eps:
        left = rdp(points[:idx+1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

# ---------- Chaikin subdivision ----------
def chaikin_subdivision(points: List[Tuple[float,float]], iterations: int = 1) -> List[Tuple[float,float]]:
    pts = [(float(x), float(y)) for (x,y) in points]
    for _ in range(max(0, iterations)):
        if len(pts) < 2:
            break
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            p0 = pts[i]; p1 = pts[i+1]
            q = (0.75*p0[0] + 0.25*p1[0], 0.75*p0[1] + 0.25*p1[1])
            r = (0.25*p0[0] + 0.75*p1[0], 0.25*p0[1] + 0.75*p1[1])
            new_pts.append(q); new_pts.append(r)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

# ---------- Catmull-Rom ----------
def catmull_rom_chain(points: List[Tuple[float,float]], samples_per_segment: int = 6) -> List[Tuple[float,float]]:
    if len(points) < 2:
        return points[:]
    pts = [(float(x), float(y)) for x,y in points]
    if len(pts) == 2:
        a,b = pts
        out = []
        for i in range(samples_per_segment+1):
            t = i / samples_per_segment
            out.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t))
        return out
    extended = [pts[0]] + pts + [pts[-1]]
    result: List[Tuple[float,float]] = []
    for i in range(len(extended)-3):
        p0 = extended[i]; p1 = extended[i+1]; p2 = extended[i+2]; p3 = extended[i+3]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t*t; t3 = t2*t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
            result.append((x,y))
    result.append((pts[-1][0], pts[-1][1]))
    return result

def smooth_path(points: List[Tuple[int,int]], chaikin_iters: int = 1, cr_samples: int = 6) -> List[Tuple[float,float]]:
    pts = [(float(x), float(y)) for x,y in points]
    if chaikin_iters > 0:
        pts = chaikin_subdivision(pts, iterations=chaikin_iters)
    pts = catmull_rom_chain(pts, samples_per_segment=max(4, cr_samples))
    return pts

# ---------- Coordinate mapping ----------
def px_to_turtle(px: float, py: float, img_w: int, img_h: int,
                 canvas_w=CANVAS_W, canvas_h=CANVAS_H, margin=40):
    scale_x = (canvas_w - margin) / img_w
    scale_y = (canvas_h - margin) / img_h
    scale = min(scale_x, scale_y)
    tx = (px - img_w/2.0) * scale
    ty = (img_h/2.0 - py) * scale
    return tx, ty, scale

# ---------- Thickness helpers ----------
def _path_curvature_measure(path: List[Tuple[float,float]]) -> float:
    if len(path) < 3:
        return 0.0
    total = 0.0
    cnt = 0
    for i in range(1, len(path)-1):
        x0,y0 = path[i-1]; x1,y1 = path[i]; x2,y2 = path[i+1]
        vx1, vy1 = x1-x0, y1-y0
        vx2, vy2 = x2-x1, y2-y1
        a1 = math.atan2(vy1, vx1); a2 = math.atan2(vy2, vx2)
        da = abs((a2 - a1 + math.pi) % (2*math.pi) - math.pi)
        total += da
        cnt += 1
    return (total / max(1, cnt))


def save_paths_to_png(paths, img_w, img_h, out_path,
                      canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                      bg_color=BG_COLOR, line_color=LINE_COLOR,
                      thickness_mode="fixed", min_width=0.3, max_width=1.6,
                      base_line_width=LINE_WIDTH):
    """
    Vẽ lại các đường (paths) lên PIL.Image và lưu PNG trực tiếp.
    Giữ logic pensize giống draw_paths_turtle (tính theo scale).
    """
    # scale mapping (lấy scale từ px_to_turtle)
    _, _, scale = px_to_turtle(0, 0, img_w, img_h, canvas_w, canvas_h)

    im = Image.new("RGB", (int(canvas_w), int(canvas_h)), bg_color)
    draw = ImageDraw.Draw(im)

    path_lengths = [len(p) for p in paths] if paths else [1]
    max_len = max(1, max(path_lengths))

    for path in paths:
        if not path:
            continue

        # compute pw (same logic as draw_paths_turtle)
        if thickness_mode == "fixed":
            pw = base_line_width
        elif thickness_mode == "length":
            L = len(path)
            t = L / max_len
            pw = min_width + (max_width - min_width) * (0.1 + 0.9 * t)
        elif thickness_mode == "curvature":
            curv = _path_curvature_measure(path)
            ncurv = min(1.0, curv / math.pi)
            pw = min_width + (max_width - min_width) * (1.0 - ncurv)
        else:
            pw = base_line_width

        # convert to pixel width for PIL (scale factor)
        pixel_w = max(1, int(round(max(0.15, pw) * scale)))

        # convert points to canvas pixel coords (origin top-left)
        tpts = [px_to_turtle(x, y, img_w, img_h, canvas_w, canvas_h)[:2] for (x,y) in path]
        # px_to_turtle returns centered coords with +x right and +y up;
        # convert to image coords (0,0) top-left
        coords = [(int(round(tx + canvas_w/2.0)), int(round(canvas_h/2.0 - ty))) for (tx,ty) in tpts]

        if len(coords) >= 2:
            draw.line(coords, fill=line_color, width=pixel_w)

    outdir = os.path.dirname(out_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    im.save(out_path, "PNG")
    print(f"Saved PNG: {out_path}")


# ---------- Draw with turtle (fast-friendly + thickness modes) ----------
def draw_paths_turtle(paths: List[List[Tuple[float,float]]],
                      img_w: int, img_h: int,
                      delay: float = 0.001, update_every: int = 30,
                      batch: int = 20, fast: bool = False,
                      line_width: float = LINE_WIDTH, line_color: str = LINE_COLOR,
                      thickness_mode: str = "fixed", min_width: float = 0.3, max_width: float = 1.6,
                      save_out: Optional[str] = None, keep_open: bool = True):
    """
    NOTE: min_width and max_width defaults reduced to make strokes thinner by default.
    save_out: when provided -> save PNG to that path (direct PIL rendering).
    keep_open: if True, call screen.mainloop() at end so window stays open until closed.
    """
    screen = turtle.Screen()
    screen.setup(CANVAS_W, CANVAS_H)
    screen.title("Portrait line-art (vẽ từng nét)")
    screen.bgcolor(BG_COLOR)
    screen.tracer(0,0)
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.color(line_color)
    pen.speed(0)

    path_lengths = [len(p) for p in paths] if paths else [1]
    max_len = max(1, max(path_lengths))

    strokes_since_update = 0
    for i, path in enumerate(paths):
        if not path:
            continue
        # choose pen size for this path
        if thickness_mode == "fixed":
            pw = line_width
        elif thickness_mode == "length":
            L = len(path)
            t = L / max_len
            pw = min_width + (max_width - min_width) * (0.1 + 0.9 * t)
        elif thickness_mode == "curvature":
            curv = _path_curvature_measure(path)
            ncurv = min(1.0, curv / math.pi)
            pw = min_width + (max_width - min_width) * (1.0 - ncurv)
        else:
            pw = line_width
        pen.pensize(max(0.15, pw))

        # convert to turtle coords
        tpts = [px_to_turtle(x,y,img_w,img_h, CANVAS_W, CANVAS_H)[:2] for (x,y) in path]
        if not tpts:
            continue
        pen.penup()
        pen.goto(tpts[0])
        pen.pendown()
        for j, pt in enumerate(tpts[1:], start=1):
            pen.goto(pt)
            if (not fast) and (j % update_every == 0):
                screen.update()
                time.sleep(delay)
        pen.penup()
        strokes_since_update += 1
        if fast:
            if strokes_since_update >= batch:
                screen.update()
                strokes_since_update = 0
        else:
            if (i+1) % 10 == 0:
                screen.update()
                time.sleep(delay)
    # final update
    screen.update()

    # --- save canvas to file if requested (direct PNG via PIL) ---
    if save_out:
        try:
            save_paths_to_png(
                paths, img_w, img_h, save_out,
                canvas_w=CANVAS_W, canvas_h=CANVAS_H,
                bg_color=BG_COLOR, line_color=line_color,
                thickness_mode=thickness_mode,
                min_width=min_width, max_width=max_width,
                base_line_width=line_width
            )
        except Exception as e:
            print("Lỗi khi lưu PNG trực tiếp:", e)

    # leave window open until closed by user (default)
    if keep_open:
        try:
            screen.mainloop()
        except Exception:
            try:
                turtle.done()
            except Exception:
                pass
    else:
        try:
            turtle.bye()
        except Exception:
            pass

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Portrait -> line-art turtle drawer (fast + smoothing + thickness modes)")
    parser.add_argument('--input', '-i', required=True, help='Path to input portrait image (PNG/JPG).')
    parser.add_argument('--blur', type=float, default=1.0, help='Gaussian blur radius before edge detection.')
    parser.add_argument('--edge_mul', type=float, default=1.0, help='Edge threshold multiplier (higher -> fewer edges).')
    parser.add_argument('--eps', type=float, default=1.2, help='RDP epsilon (simplify).')
    parser.add_argument('--chaikin', type=int, default=1, help='Chaikin subdivision iterations (0 = off).')
    parser.add_argument('--cr_samples', type=int, default=6, help='Catmull-Rom samples per segment (higher = smoother).')
    parser.add_argument('--min_path_len', type=int, default=0, help='Discard paths shorter than this (speedup / denoise).')
    parser.add_argument('--maxsize', type=int, nargs=2, default=(700,700), help='Max image size (W H).')
    parser.add_argument('--delay', type=float, default=0.001, help='Delay between partial updates when not in fast mode.')
    parser.add_argument('--update_every', type=int, default=30, help='Update every N points when not in fast mode.')
    parser.add_argument('--batch', type=int, default=20, help='When --fast: update screen after this many strokes.')
    parser.add_argument('--fast', action='store_true', help='Fast drawing mode: minimal updates and no sleeps.')
    parser.add_argument('--line_color', type=str, default=LINE_COLOR, help='Line color (turtle).')
    parser.add_argument('--line_width', type=float, default=LINE_WIDTH, help='Base pen width for drawing.')
    parser.add_argument('--thickness_mode', choices=['fixed','length','curvature'], default='fixed',
                        help='How to vary stroke thickness: fixed/length/curvature.')
    parser.add_argument('--min_width', type=float, default=0.3, help='Min pen width when using dynamic thickness.')
    parser.add_argument('--max_width', type=float, default=1.6, help='Max pen width when using dynamic thickness.')
    parser.add_argument('--save_out', type=str, default=None, help='Path to save PNG output (e.g. data/output/result.png).')
    parser.add_argument('--no_keep', action='store_true', help='Do not keep window open after drawing (close immediately).')
    args = parser.parse_args()

    img = load_image_gray(args.input, max_size=tuple(args.maxsize))
    print("Loaded image size:", img.size)
    print(f"Detecting edges (blur={args.blur:.2f}, edge_mul={args.edge_mul:.2f})...")
    edges = sobel_edges(img, blur_radius=args.blur, edge_mul=args.edge_mul)
    print("Edge pixels:", int(edges.sum()))
    print("Tracing connected paths...")
    paths = find_connected_paths(edges)
    print("Found paths:", len(paths))

    # filter short paths if requested
    if args.min_path_len > 0:
        before = len(paths)
        paths = [p for p in paths if len(p) >= args.min_path_len]
        print(f"Filtered short paths (<{args.min_path_len}): {before} -> {len(paths)}")

    # simplify + smooth
    final_paths: List[List[Tuple[float,float]]] = []
    for p in paths:
        sp = rdp(p, eps=args.eps)
        if len(sp) < 2:
            continue
        sm = smooth_path(sp, chaikin_iters=args.chaikin, cr_samples=args.cr_samples)
        final_paths.append(sm)
    print("Paths after simplify + smoothing:", len(final_paths))

    # prepare save_out default if user didn't provide but wants automatic naming
    save_out = args.save_out
    if save_out is None:
        inp_base = os.path.basename(args.input)
        name, _ = os.path.splitext(inp_base)
        save_out = os.path.join("data", "output", f"{name}.png")

    print("Drawing with turtle... (window will open) — saving to:", save_out)
    draw_paths_turtle(final_paths, img.width, img.height,
                      delay=args.delay, update_every=args.update_every,
                      batch=args.batch, fast=args.fast,
                      line_width=max(0.15, args.line_width),
                      line_color=args.line_color,
                      thickness_mode=args.thickness_mode,
                      min_width=max(0.05, args.min_width),
                      max_width=max(0.05, args.max_width),
                      save_out=save_out,
                      keep_open=(not args.no_keep))

if __name__ == '__main__':
    main()

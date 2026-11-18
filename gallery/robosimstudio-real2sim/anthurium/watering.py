"""Water with the catalog hand: all cup motion is through pad contact."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robosimstudio_studio.routine import reach, grip, wait, hand

ROOT = Path(__file__).resolve().parent
ORIGIN = np.array([0.6,0,0.75])
LIP = np.array([0,0,0.078])


def readings(sim):
    points = sim['water'].particle_q.astype(float) - ORIGIN
    in_pot = (np.linalg.norm(points[:,:2],axis=1) < 0.063) & (points[:,2] > 0.180) & (points[:,2] < 0.216)
    pose = np.asarray(sim['cup'].pose)
    local = (points+ORIGIN-pose[:3]) @ Rotation.from_quat(pose[3:]).as_matrix()
    in_cup = (np.abs(local[:,:2]).max(axis=1) < 0.026) & (local[:,2] > 0.006) & (local[:,2] < 0.08)
    return dict(particles=len(points), in_pot_pct=float(100*in_pot.mean()), in_cup_pct=float(100*in_cup.mean()),
                on_table_pct=float(100*((points[:,2] < 0.08)&(points[:,2]>-0.01)&~in_cup&~in_pot).mean()),
                below_table_pct=float(100*(points[:,2] < -0.01).mean()), cup_pose=pose.tolist())


def _water(sim, session):
    hz = float(sim.request.physics.control_hz)
    panel = session.arms['right']
    cup = sim['cup']
    rest = {f'leaf_{i:02d}': sim[f'leaf_{i:02d}'].particle_q.copy() for i in range(1,13)}
    peaks = {n:0.0 for n in rest}
    tip_ids = {}
    # The free point farthest from the held basal patch is the geometric tip.
    for n,p in rest.items():
        held = sim[n].entity.held
        base = sim.state.particle_q.numpy()[held].mean(0)
        tip_ids[n] = int(np.linalg.norm(p-base,axis=1).argmax())
    tip_peaks = {n:0.0 for n in rest}
    samples = []
    events = []
    impulse_start = float(sim.pipeline.solver.contact_impulse.numpy()[0])
    def mark(name):
        item = dict(name=name,time=sim.time,**readings(sim))
        events.append(item)
        (ROOT/'run'/'watering_events.json').write_text(json.dumps(events,indent=2))
        print(name, 'pot', round(item['in_pot_pct'],1), 'cup', round(item['in_cup_pct'],1), flush=True)
    def observe():
        for n, p in rest.items():
            delta = np.linalg.norm(sim[n].particle_q-p,axis=1)*1000
            peaks[n] = max(peaks[n], float(delta.max()))
            tip_peaks[n] = max(tip_peaks[n], float(delta[tip_ids[n]]))
        if sim.step_count % 6 == 0:
            samples.append(dict(time=sim.time,tip_mm={n:float(np.linalg.norm(sim[n].particle_q[tip_ids[n]]-p[tip_ids[n]])*1000) for n,p in rest.items()}, impulse=float(sim.pipeline.solver.contact_impulse.numpy()[0])-impulse_start))
    # Use the old take's upright grasp orientation, computed with today's arm.
    seed = sim.state.joint_q.numpy().copy()
    seed[panel.ik.arm_coords] = (0,1.723,0.642,1.081,0,0)
    _, upright = panel.ik.fk_ee(seed)
    upright = (Rotation.from_euler("z",60,degrees=True)*Rotation.from_quat(upright)).as_quat()
    standing = np.asarray(cup.position)
    # Unfold outside the bench edge before crossing above the work surface.
    panel.suspend()
    qpark=sim.state.joint_q.numpy()[panel.ik.arm_coords].copy()
    qready=np.array([-1.57,0,0,0,0,0])
    for k in range(1,76):
        u=k/75;u=u*u*(3-2*u)
        panel.ik.set_joint_targets(qpark+(qready-qpark)*u,sim.control.joint_target_q)
        yield
    panel.resume()
    yield from wait(session,.1)
    yield from reach(session,'right',standing+[0,0,0.18],quat=upright,speed=0.4,turn=4,settle=.9,tolerance=0.012)
    yield from reach(session,'right',standing+[0,0,0.055],quat=upright,speed=0.20,turn=4,settle=.6,tolerance=0.008)
    yield from grip(session,'right',1.0,seconds=0.55)
    mark('grasped')
    yield from reach(session,'right',standing+[0,0,0.43],quat=upright,speed=0.18,turn=3,settle=1,tolerance=0.035)
    if cup.position[2] < ORIGIN[2]+0.30:
        raise RuntimeError('The cup did not lift in the fingers')
    mark('lifted')
    panel.suspend()
    ik_seed = sim.state.joint_q.numpy().copy()
    tcp, quat = hand(session,'right')
    q0 = Rotation.from_quat(quat)
    pose = np.asarray(cup.pose)
    lip_now = lambda: np.asarray(cup.pose)[:3] + Rotation.from_quat(np.asarray(cup.pose)[3:]).apply(LIP)
    offset = lip_now()-tcp
    lip = lip_now()-ORIGIN
    angle = 0.0
    bias = np.zeros(3)
    azimuth = 70.0
    # Carry above the spadices, then lower the side-facing opening over the free blade.
    plan = [
        ('carry', (-0.15, -0.17, 0.6), 0, 1.5, 70),
        ('lean', (-0.15, 0.14, 0.6), 65, 2.0, 70),
        ('above_leaf', (-0.03, 0.195, 0.54), 81, 1.2, 70),
        ('on_leaf', (-0.03, 0.195, 0.54), 85, 1.8, 70),
        ('stop_stream', (-0.15, 0.14, 0.6), 65, 0.8, 70),
        ('withdraw_leaf', (-0.15, -0.17, 0.6), 0, 1.3, 70),
        ('clear_canopy', (-0.14, -0.15, 0.55), 0, 0.4, 70),
        ('pre_pot', (-0.14, -0.15, 0.62), 65, 0.8, 70),
        ('over_pot', (0.05, -0.03, 0.595), 65, 1, 70),
        ('pour', (0.05, -0.03, 0.595), 130, 2.0, 70),
        ('drain', (0.05, -0.03, 0.595), 130, 2, 70),
        ('lift_out', (-0.075, -0.08, 0.59), 120, 0.8, 70),
        ('back', (-0.12, -0.17, 0.5), 65, 0.8, 70),
        ('righting', (-0.08, -0.17, 0.48), 0, 1.2, 70),
        ('retreat', (-0.14, -0.21, 0.38), 0, 1.3, 70),
    ]
    try:
        for name,target,tilt,seconds,az in plan:
            target = np.asarray(target)
            for k in range(1,round(seconds*hz)+1):
                u = k/round(seconds*hz); t = u*u*(3-2*u)
                aim = lip+(target-lip)*t
                a = angle+(tilt-angle)*t
                bearing=np.deg2rad(azimuth+(az-azimuth)*t)
                axis=-np.array([np.cos(bearing),np.sin(bearing),0])
                rot = Rotation.from_rotvec(axis*np.deg2rad(a))
                bias += 0.02*(ORIGIN+aim-lip_now())
                size = np.linalg.norm(bias)
                if size > 0.015: bias *= 0.015/size
                target_pos = ORIGIN+aim-rot.apply(offset)+bias
                target_quat = (rot*q0).as_quat()
                ik_seed = panel.ik.solve_q_full(ik_seed,target_pos,target_quat,iterations=60)
                solved,_ = panel.ik.fk_ee(ik_seed)
                if np.linalg.norm(solved-target_pos) > 0.012:
                    panel.resume()
                    raise RuntimeError(f"Pour pose is unreachable: {name}, {np.linalg.norm(solved-target_pos)*1000:.1f} mm")
                panel.ik.set_joint_targets(ik_seed[panel.ik.arm_coords],sim.control.joint_target_q)
                yield
                observe()
            lip,angle,azimuth = target,tilt,az
            mark(name)
    finally:
        panel.resume()
    panel.dragging = False
    yield from wait(session,2.0)
    observe()
    result = dict(**readings(sim), peak_leaf_displacement_mm=peaks, peak_tip_displacement_mm=tip_peaks,
        tip_ids=tip_ids, tip_samples=samples, water_contact_impulse_Ns=float(sim.pipeline.solver.contact_impulse.numpy()[0])-impulse_start,
        residual_tip_displacement_mm={n:float(np.linalg.norm(sim[n].particle_q[tip_ids[n]]-p[tip_ids[n]])*1000) for n,p in rest.items()},
        residual_leaf_displacement_mm={n:float(np.linalg.norm(sim[n].particle_q-p,axis=1).max()*1000) for n,p in rest.items()}, events=events)
    (ROOT/'run'/'watering_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("tip_samples", "events")},indent=2),flush=True)
    return result


def routine(sim, session):
    """Run the physical chore and retain states for a second camera's film."""
    frames = []
    result = None
    stamp = time.time_ns()
    def capture(env):
        if env.step_count % 6 == 0:
            frames.append((env.time, env.state.body_q.numpy().copy(), env.state.particle_q.numpy().copy()))
    capture(sim)
    sim.post_step_hooks.append(capture)
    try:
        result = yield from _water(sim, session)
    finally:
        if capture in sim.post_step_hooks:
            sim.post_step_hooks.remove(capture)
        if frames:
            np.savez_compressed(ROOT/'run'/'last_run_states.npz',
                time=np.array([f[0] for f in frames]),body_q=np.stack([f[1] for f in frames]),
                particle_q=np.stack([f[2] for f in frames]))
        archive = ROOT/'run'/'replays'
        archive.mkdir(exist_ok=True)
        summary = result if result is not None else dict(**readings(sim), incomplete=True, time=sim.time)
        (archive/f'{stamp}.json').write_text(json.dumps(summary,indent=2))

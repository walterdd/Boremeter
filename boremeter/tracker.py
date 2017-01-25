from bbs import *


def track_faces(bbs, timeout=200):
    
    faces = {}
    max_id = 1

    for fn in bbs:
        faces[fn] = [0, {}]
        faces[fn][0] = bbs[fn][0]
        for bb in bbs[fn][1]:
            faces[fn][1][max_id] = {}
            faces[fn][1][max_id]['timeout'] = timeout
            faces[fn][1][max_id]['coords'] = bb
            max_id += 1
            
    
    new_faces = {}
    new_faces[0] = faces[0]
    
    for frame in faces:
        if frame != 0:
            new_faces[frame] = [0, {}]
            new_faces[frame][0] = bbs[frame][0]
            for bb in faces[frame][1]:
                found = 0
                intersect = 0
                for prev_id in new_faces[frame - 1][1]:          
                    if have_intersection(new_faces[frame - 1][1][prev_id]['coords'],
                                         faces[frame][1][bb]['coords']):
                        intersect = 1
                        
                    if new_faces[frame - 1][1][prev_id]['timeout'] > 0:
                        if not found and are_close(new_faces[frame - 1][1][prev_id]['coords'], 
                                                   faces[frame][1][bb]['coords']):
                            new_faces[frame][1][prev_id] = faces[frame][1][bb]
                            new_faces[frame - 1][1][prev_id]['timeout'] = -1
                            new_faces[frame][1][prev_id]['timeout'] = timeout
                            found = 1
                
                if not found and not intersect:
                    new_faces[frame][1][bb] = faces[frame][1][bb]
                            
            for prev_id in new_faces[frame - 1][1]:
                if new_faces[frame - 1][1][prev_id]['timeout'] > 0:
                    new_faces[frame - 1][1][prev_id]['timeout'] -= 1
                    new_faces[frame][1][prev_id] = new_faces[frame - 1][1][prev_id]

    return new_faces


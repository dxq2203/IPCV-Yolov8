from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTranformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np

def main():
    #Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')


    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    #Get object positions
    tracker.add_position_to_tracks(tracks)
    
    #Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTranformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    #Interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks["ball"])
    
    #Speed and Distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    #Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = [] # for tracking time keeping ball 
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assign_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assign_player != -1:
            assigned_player = tracks['players'][frame_num][assign_player]
            assigned_player['has_ball'] = True
            team_ball_control.append(assigned_player['team'])
        else:
            # the last player who has the ball
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control) #convert to numpy array  
    
    #Draw output
    #Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(video_frames, tracks)
    #Save video
    save_video(output_video_frames, 'output_videos/output_videos.avi')
    
if __name__ == "__main__":
    main()

# idea train the best model

# idea of predict movement: the bounding box of the previous and the next frame is different
# because the x,y,x,y inspite a same entity 
# by tracking we get the position of the entity, we predict the next movement of the object base on
# from the closest and the current frame
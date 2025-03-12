from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')


    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    #draw output
    
    #Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save video
    save_video(output_video_frames, 'output_videos/output_videos.avi')
    
if __name__ == "__main__":
    main()

# idea train the best model

# idea of predict movement: the bounding box of the previous and the next frame is different
# because the x,y,x,y inspite a same entity 
# by tracking we get the position of the entity, we predict the next movement of the object base on
# from the closest and the current frame
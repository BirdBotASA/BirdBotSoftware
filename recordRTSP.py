from threading import Thread
import cv2, time, subprocess

counter = 0
captureAudio = False

class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        
        # Create a VideoCapture object
        if captureAudio == True:
            self.cap_audio = subprocess.Popen('vlc --novideo -Idummy ' + str(src))
        
        time.sleep(1)
        
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('m','p','4','v')
        self.output_video = cv2.VideoWriter('output.mp4', self.codec, 20, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            self.frame = cv2.resize(self.frame, (1920, 1080))
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            if captureAudio == True:
                self.cap_audio.terminate()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        # print("save - " + str(counter))
        self.output_video.write(self.frame)

rtsp_stream_link = 'rtsp://ubnt:ubnt@162.251.9.138:554/s2'
video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)

while True:
    
    times, times_2 = [], []
    t = time.localtime()
    t1 = time.time()
    
    try:
        video_stream_widget.show_frame()
        video_stream_widget.save_frame()
    except AttributeError:
        pass
    
    time.sleep(.000001)
    
    t2 = time.time()
    
    t3 = time.time()
    times.append(t2-t1)
    times_2.append(t3-t1)

    times = times[-60:]
    times_2 = times_2[-60:]

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
    
    # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
    
    counter += 1
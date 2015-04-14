package org.opencv.samples.colorblobdetect;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.R;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.widget.MediaController;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.View.OnTouchListener;
import android.widget.VideoView;

public class ColorBlobDetectionActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {
	private static final String  TAG              = "OCVSample::Activity";

	private boolean              mIsColorSelected = false;
	private Mat                  mRgba;
	private Scalar               mBlobColorRgba;
	private Scalar               mBlobColorHsv;
	private ColorBlobDetector    mDetector;
	private Mat                  mSpectrum;
	private Size                 SPECTRUM_SIZE;
	private Scalar               CONTOUR_COLOR;
	List<Mat> splitted = new ArrayList<Mat>();
	Mat onlyRed;
	Scalar lower; 
	Scalar upper; 
	Mat templateImage;
	Mat resultImage;
	int cameraWidth;
	int cameraHeight;
	private long frameCount;
	ExecutorService executor;
	FutureTask future;
	Rect cropRect;
	MatOfDouble mn1;
	MatOfDouble stdev1;
	MatOfDouble mn2;
	MatOfDouble stdev2;
	private CameraBridgeViewBase mOpenCvCameraView;

	private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS:
			{
				Log.i(TAG, "OpenCV loaded successfully");
				mOpenCvCameraView.enableView();
				mOpenCvCameraView.setOnTouchListener(ColorBlobDetectionActivity.this);
			} break;
			default:
			{
				super.onManagerConnected(status);
			} break;
			}
		}
	};

	public ColorBlobDetectionActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(org.opencv.samples.colorblobdetect.R.layout.color_blob_detection_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(org.opencv.samples.colorblobdetect.R.id.color_blob_detection_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onPause()
	{
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume()
	{
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		mDetector = new ColorBlobDetector();
		mSpectrum = new Mat();
		mBlobColorRgba = new Scalar(255);
		mBlobColorHsv = new Scalar(255);
		SPECTRUM_SIZE = new Size(200, 64);
		CONTOUR_COLOR = new Scalar(255,0,0,255);
		cameraWidth = width;
		cameraHeight = height;
		Log.w("camerasize",width+":"+height);
		executor = Executors.newFixedThreadPool(2);		

		//		future = new FutureTask(new Callable() {
		//					public String call() {
		//						return "";
		//					};
		//				});

		onlyRed = new Mat();
		resultImage = new Mat(CvType.CV_32F);
		lower = new Scalar(100);
		upper = new Scalar(200);	

		mn1 = new MatOfDouble();
		stdev1 = new MatOfDouble();
		mn2 = new MatOfDouble();
		stdev2 = new MatOfDouble();

		cropRect = new Rect(new Point(169,22),new Point(630,457));
		//org.opencv.samples.colorblobdetect.
		File root = Environment.getExternalStorageDirectory();
		String rootPath= root.getPath();
		
		
		//Bitmap bMap=BitmapFactory.decodeFile("segmentierung.png");
		BitmapFactory.Options options = new BitmapFactory.Options();
		options.outHeight = 435;
		options.outWidth = 461;
		options.inScaled = false;
		
		
		Bitmap bMap = BitmapFactory.decodeResource(getResources(),org.opencv.samples.colorblobdetect.R.drawable.segmentierung,options);
		//bMap.reconfigure(800, 400, Config.ARGB_8888);
		templateImage = new Mat ( 461,435, CvType.CV_8U, new Scalar(4));
		Bitmap myBitmap32 = bMap.copy(Bitmap.Config.ARGB_8888, true);
		Utils.bitmapToMat(myBitmap32, templateImage);
		Log.w("themat","mat:"+templateImage);
		Imgproc.resize(templateImage, templateImage, new Size(461,435));
		Imgproc.cvtColor(templateImage, templateImage, Imgproc.COLOR_BGRA2GRAY);

		//Imgproc.Canny(templateImage, templateImage, 100, 200);
		List<Mat> channels = new ArrayList<Mat>();

		Core.split(templateImage,channels);
		//Collections.reverse(channels);
		Core.merge(channels,templateImage);
		
		double meanTemplateImage = Core.mean(templateImage).val[0];
		for(int i=0; i < templateImage.rows(); ++i)
		{
			for(int j=0; j < templateImage.cols(); ++j)
			{
				double[] values = templateImage.get(i, j);
				double val = values[0]-meanTemplateImage;
				val =  val > 0 ? val : 0;
				
				templateImage.put(i,j,val);
			}
		}
		
		templateImage.convertTo(templateImage, CvType.CV_8U);
		//Imgproc.resize(src, dst, dsize);
	}

	public void onCameraViewStopped() {
		mRgba.release();
	}

	public boolean onTouch(View v, MotionEvent event) {
		return false; // don't need subsequent touch events
	}
	double maxCC = Double.MIN_VALUE;
	Mat cropped;
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		mRgba = inputFrame.rgba();

		//mRgba = mRgba.adjustROI(cropRect.y,cropRect.y+cropRect.height,cropRect.x,cropRect.x+cropRect.width);
		Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2GRAY);
		cropped = new Mat(mRgba,cropRect);
		//Imgproc.Canny(mRgba,mRgba, 100, 200);
		long t0 = System.currentTimeMillis();
		
		double meanCropped = Core.mean(cropped).val[0];
		for(int i=0; i < cropped.rows(); ++i)
		{
			for(int j=0; j < cropped.cols(); ++j)
			{
				
				double[] values = cropped.get(i, j);
				double val = values[0]-meanCropped;
				val = val > 0 ? val : 0;
				cropped.put(i,j,val);
			}
		}
		cropped.convertTo(cropped, CvType.CV_8U);
		
		
		Imgproc.matchTemplate(cropped, templateImage, resultImage,Imgproc.TM_CCORR_NORMED);
		Log.w("time","dt: "+(System.currentTimeMillis()-t0));
		//Core.split(mRgba, splitted);
		//Core.inRange(splitted.get(0),lower, upper,onlyRed);

		String ccResult = "cc: ";
		Log.w("templateimage",templateImage.width()+":"+templateImage.height());
		Log.w("resultsize",resultImage.width()+":"+resultImage.height());
		double cc = 0;
		if(resultImage.width()==1&&resultImage.height()==1){
			cc = resultImage.get(0, 0)[0];
		}			

		Log.w("chs",mRgba.channels()+" "+templateImage.channels());
		//Core.addWeighted(mRgba, 0.7,templateImage,0.3,0,mRgba);
		//double cc = normalizedCrossCorrelation(cropped, templateImage.clone());
		
		if(cc>maxCC){
			maxCC = cc;
		}
		
		Core.putText(mRgba, "cc: "+String.format("%.3f",cc),new Point(100,100),Core.FONT_HERSHEY_PLAIN,
				3,new Scalar(255,255,255,255));
		Core.putText(mRgba, "cc max: "+String.format("%.3f",maxCC),new Point(100,130),Core.FONT_HERSHEY_PLAIN,
				3,new Scalar(255,255,255,255));
		Core.rectangle(mRgba, cropRect.tl(),cropRect.br(),new Scalar(0,0,255,255));
		//executor.execute(future);
		return mRgba;
	}
	 
	private double normalizedCrossCorrelation(Mat im1,Mat im2)
	{		 
		
		im1.convertTo(im1, CvType.CV_64F);
		im2.convertTo(im2, CvType.CV_64F);
		
		//Scalar mean1 = Core.mean(im1);
		//Scalar mean2 = Core.mean(im2);

		double wert = 0;
		double nenner1 = 0;
		double nenner2 = 0;
		double nenner = 0;
		 
		 im1 = im1.reshape(1,im1.rows()*im1.cols());
		 im2 = im2.reshape(1,im2.rows()*im2.cols());

		 double c1;
		 double c2;
		 
		 double sigma1;
		 double sigma2;
		 
		 Core.meanStdDev(im1, mn1, stdev1);
		 
		 Core.meanStdDev(im2, mn2, stdev2);
		 
		 double mean1 = mn1.get(0, 0)[0];
		 double mean2 = mn2.get(0, 0)[0];
		 
		 double std1 = stdev1.get(0, 0)[0];
		 double std2 = stdev2.get(0, 0)[0];
		 
		  for(int i=0; i < im1.rows();++i)
		  {
			  
			  c1 = im1.get(i, 0)[0]-mean1;
			  c2 = im2.get(i, 0)[0]-mean2;
			  
			  wert += (c1 * c2) / (std1 * std2);
		  }
		  
//		  
//		  for(int i=0; i < im1.rows();++i)
//		  {
//			  c1 = im1.get(i, 0)[0]-mean1.val[0];
//			  nenner1 += c1 * c1;
//		  }
//		  for(int i=0; i < im2.rows();++i)
//		  {
//			  c2 = im2.get(i, 0)[0]-mean2.val[0];
//			  nenner2 += c2 * c2;
//		  }

//
//		  nenner = nenner1 * nenner2;
//		  nenner = Math.sqrt(nenner);     
//		 
          int rows = im1.rows();
          
		  return wert / im1.rows();
}

	//	public void playVideo(){
	//		VideoView videoView = (VideoView)this.findViewById(R.id.videoView);
	//	      MediaController mc = new MediaController(this);
	//	      videoView.setMediaController(mc);
	//
	//	        videoView.setVideoPath(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES) +"/movie.mp4");
	//
	//	 
	//	      videoView.requestFocus();
	//	      videoView.start();
	//	}
	public boolean isExternalStorageWritable() {
	    String state = Environment.getExternalStorageState();
	    if (Environment.MEDIA_MOUNTED.equals(state)) {
	        return true;
	    }
	    return false;
	}

	/* Checks if external storage is available to at least read */
	public boolean isExternalStorageReadable() {
	    String state = Environment.getExternalStorageState();
	    if (Environment.MEDIA_MOUNTED.equals(state) ||
	        Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
	        return true;
	    }
	    return false;
	}

}

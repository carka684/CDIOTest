import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.HOGDescriptor;

import edu.wildlifesecurity.framework.identification.impl.HOGIdentification;
import edu.wildlifesecurity.framework.identification.impl.ImageReader;



public class hello
{
	public static void main( String[] args )
	{
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		//testMat();
		HOGIdentification hogTest = new HOGIdentification(); 
		hogTest.init();
		hogTest.trainClassifier("Images/trainIm/","", "SVM.txt");
		hogTest.evaluateClassifier("Images/valIm/", "");
//		ImageReader reader = new ImageReader();
//		reader.readImages("Images/trainIm/","");
		//testMat2();
	}
	public static void testMat2()
	{
		Mat matrix = Mat.ones(new Size(5,5), CvType.CV_16S);
		Mat row = Mat.ones(new Size(1,5), CvType.CV_16S);
		Mat submat = matrix.submat(0, 1, 0, 5);
		Core.add(submat, row.t(), submat);
		System.out.println(matrix.dump());
	}
	
	public static Vector<String> listFilesForFolder(String classFolder) {
		Vector<String> filesVec = new Vector<String>();
		File[] folders = new File(classFolder).listFiles();
		int classNum = 0;
		Map<Integer, Vector<String>> map = new HashMap<Integer, Vector<String>>();
		for (File folder : folders) {
		    if (folder.isDirectory()) {
		    	map.put(classNum++,getFiles(folder));
		    }
		}
		System.out.println(map.get(2).size());
		return filesVec;
	}
	public static Vector<String> getFiles(File folder)
	{		
		Vector<String> filesVec = new Vector<String>();
		for (File file : folder.listFiles()) {
		    if (file.isFile()) {
		    	filesVec.add(folder + "\\" + file.getName());
		    }
		}
		return filesVec;
	}
	
	public static void testMat()
	{
		double[] classesArray = {0,0,0,0,1,1,1,2,2,2};
		double[] resultArray =  {0,0,1,0,1,0,0,2,0,0};
		int[] numPerClass = {4,3,3};
		int numOfClasses = 3;
		Mat classes = new Mat(classesArray.length,1,CvType.CV_32F);
		Mat results = new Mat(resultArray.length,1,CvType.CV_32F);
		classes.put(0, 0, classesArray);
		results.put(0, 0, resultArray);
		int pos = 0;
		for(int i = 0; i < numOfClasses; i++)
		{
			
			Mat temp = results.submat(pos, pos+numPerClass[i], 0, 1).clone();
			pos += numPerClass[i];
			Core.subtract(temp, new Scalar(i), temp);
			int numOfTP = numPerClass[i] - Core.countNonZero(temp);
			Core.subtract(results,new Scalar(i),temp);
			int numOfFP = (classes.rows() - Core.countNonZero(temp)) - numOfTP;
			System.out.println("TP: " + (double) numOfTP/numPerClass[i]);
			System.out.println("FP: " + (double) numOfFP/(classes.rows()-numPerClass[i]));
		}
	}
	public static  double[] getResult(Mat classes, Mat result,int numberOfPos, int numberOfNeg)
	{
		
		Mat falseNegMat = new Mat();
		Mat falsePosMat = new Mat();
		Core.absdiff(classes.rowRange(0, numberOfPos),result.rowRange(0, numberOfPos),falseNegMat);
		Core.absdiff(classes.rowRange(numberOfPos,numberOfPos+numberOfNeg),result.rowRange(numberOfPos, numberOfPos+numberOfNeg),falsePosMat);
		
		Scalar falseNegRes =  Core.sumElems(falseNegMat);
		Scalar falsePosRes =  Core.sumElems(falsePosMat);
		double FN = falseNegRes.mul(new Scalar((double) 1/(2*numberOfPos))).val[0];
		double TP = 1-FN;
		double FP = falsePosRes.mul(new Scalar((double) 1/(2*numberOfPos))).val[0];
		double TN  = 1 - FP;
		double[] res = {TP,FN,TN,FP};
		return res;
	}
	public static Mat getDescriptors(Vector<String> strVec,HOGDescriptor hog,Size s)
	{
		
		MatOfFloat descriptors = new MatOfFloat();
		Mat m;
		Mat featMat = new Mat();
		for(String file : strVec)
		{
			m=Highgui.imread(file,Highgui.CV_LOAD_IMAGE_GRAYSCALE);
			Imgproc.resize(m, m, s);
			long startTime = System.nanoTime();
			hog.compute(m, descriptors);			
			long endTime = System.nanoTime();
			System.out.println((endTime-startTime)/1000000);
			featMat.push_back(descriptors.t());
		}
		return featMat;
	}
	public static void testSVM()
	{
		double[][] bufferTrain = {{3,2},{1,2},{0,1},{2,0},{10,11},{8,11},{10,8},{9,8},{8,-11},{10,-8},{9,-8}};
		double[] bufferClasses = {1,1,1,1,2,2,2,2,3,3,3};

		Mat samples = new Mat(bufferTrain.length,2,CvType.CV_32F);
		for(int i = 0; i < 8; i++)
		{
			samples.put(i, 0, bufferTrain[i]);
		}
		System.out.println(samples.dump());

		Mat classes = new Mat(bufferClasses.length,1,CvType.CV_32F);
		classes.put(0, 0, bufferClasses);
		System.out.println(classes.dump());
		
		CvSVM SVM = new CvSVM();
		CvSVMParams params = new CvSVMParams();
	    params.set_kernel_type(CvSVM.LINEAR);
		SVM.train(samples,classes,new Mat(),new Mat(),params);
		Mat val = new Mat(1,2,CvType.CV_32F);
		double[] value = {10,-10};
		val.put(0, 0, value);
		System.out.println(val.dump());
		System.out.println("Result: " + SVM.predict(val));
		SVM.save("test.txt");		
	}
}
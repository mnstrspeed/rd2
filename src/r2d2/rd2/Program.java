package r2d2.rd2;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.DistanceMeasure;
import r2d2.rd2.distances.EuclideanDistance;
import r2d2.rd2.distances.WeightedEuclideanDistance;

public class Program
{	
	public static void main(String[] args)
	{
		String action = args[0];
		if (action.equals("classify"))
		{
			// classify <train set path> <test set path>
			classify(args[1], args[2]);
		}
		else if (action.equals("gym"))
		{
			// gym <train set path>
			gym(args[1]);
		}
	}
	
	/**
	 * Classify a test set based on a train set
	 * @param trainPath
	 * @param testPath
	 */
	public static void classify(String trainPath, String testPath)
	{
		List<Classification<AttributeVector, Integer>> trainingSet;
		List<AttributeVector> testSet;
		
		// Read train and test set
		try
		{
			trainingSet  = read(trainPath, scanner -> new Classification<AttributeVector, Integer>(
							AttributeVector.fromScanner(scanner, 8), scanner.nextInt()));
			testSet = read(testPath, scanner -> AttributeVector.fromScanner(scanner, 8));
		}
		catch (FileNotFoundException ex)
		{
			throw new RuntimeException("File not found", ex);
		}
		
		DistanceMeasure<AttributeVector> noiseDistanceMeasure = new EuclideanDistance(); // distance measure for ENN
		DistanceMeasure<AttributeVector> classDistanceMeasure = new WeightedEuclideanDistance(new double[] { // distance measure for classification
				0, 1, 0, 1,
				0, 1, 1, 1 // weights to use
		});
		
		// Run classifier
		R2D2Classifier<AttributeVector, Integer> classifier = new R2D2Classifier<AttributeVector, Integer>(11, 
				classDistanceMeasure, noiseDistanceMeasure);
		classifier.train(trainingSet);

		List<Classification<AttributeVector, Integer>> result = classifier.classify(testSet);
		
		// Save prototypes and class labels for upload
		try
		{
			long time = System.currentTimeMillis();
			String prototypePath = "prototypes_" + time;
			String labelPath = "labels_" + time;
			
			write(prototypePath, classifier.getPrototypes(), c -> c.getDataPoint() + " " + c.getClassLabel());
			write(labelPath, result, c -> c.getClassLabel().toString());
		}
		catch (IOException ex)
		{
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * Where we train weights, duh
	 * @param trainPath
	 */
	public static void gym(String trainPath)
	{
		// TODO
	}
	
	private static <T> List<T> read(String filePath, Function<Scanner, T> reader) throws FileNotFoundException
	{
		List<T> set = new ArrayList<T>();
		
		Scanner scanner = new Scanner(new FileInputStream(filePath));
		while (scanner.hasNext())
		{
			set.add(reader.apply(scanner));
		}
		scanner.close();
		return set;
	}
	
	private static <T> void write(String filePath, List<T> elements, Function<T, String> writer) throws IOException
	{
		try (PrintWriter output = new PrintWriter(new FileWriter(filePath)))
		{
			for (T element : elements)
			{
				output.println(writer.apply(element));
			}
		}
	}
}

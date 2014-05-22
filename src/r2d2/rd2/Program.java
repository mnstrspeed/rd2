package r2d2.rd2;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
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
			System.out.print("Loading " + trainPath + "...");
			trainingSet  = read(trainPath, scanner -> new Classification<AttributeVector, Integer>(
							AttributeVector.fromScanner(scanner, 8), scanner.nextInt()));
			System.out.println(" done");
			System.out.print("Loading " + testPath + "...");
			testSet = read(testPath, scanner -> AttributeVector.fromScanner(scanner, 8));
			System.out.println(" done");
		}
		catch (FileNotFoundException ex)
		{
			throw new RuntimeException("File not found", ex);
		}
		
		long startTime = System.currentTimeMillis();
		
		DistanceMeasure<AttributeVector> noiseDistanceMeasure = new EuclideanDistance(); // distance measure for ENN
		DistanceMeasure<AttributeVector> classDistanceMeasure = new WeightedEuclideanDistance(new double[] { // distance measure for classification
				0, 1, 0, 1,
				0, 1, 1, 1 // weights to use
		});
		
		// Run classifier
		System.out.print("Training classifier...");
		R2D2Classifier<AttributeVector, Integer> classifier = new R2D2Classifier<AttributeVector, Integer>(11, 
				classDistanceMeasure, noiseDistanceMeasure);
		classifier.train(trainingSet);
		System.out.println(" done");

		System.out.print("Classifying test set...");
		List<Classification<AttributeVector, Integer>> result = classifier.classify(testSet);
		System.out.println(" done");
		
		// Save prototypes and class labels for upload
		try
		{
			String time = new SimpleDateFormat("MMMdd_HH.mm.ss").format(Calendar.getInstance().getTime());
			String prototypePath = "prototypes_" + time + ".txt";
			String labelPath = "labels_" + time + ".txt";
			
			write(prototypePath, classifier.getPrototypes(), c -> c.getDataPoint() + " " + c.getClassLabel());
			System.out.println("Saved prototypes as " + prototypePath);
			write(labelPath, result, c -> c.getClassLabel().toString());
			System.out.println("Saved labels as " + labelPath);
		}
		catch (IOException ex)
		{
			throw new RuntimeException(ex);
		}
		
		System.out.println("\nExecution took " + (System.currentTimeMillis() - startTime) + " ms");
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

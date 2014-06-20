package r2d2.rd2;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.function.Function;

import r2d2.rd2.classifier.AttributeVector;
import r2d2.rd2.classifier.Classification;
import r2d2.rd2.classifier.KNearestNeighborClassifier;
import r2d2.rd2.classifier.R2D2Classifier;
import r2d2.rd2.distances.DistanceMeasure;
import r2d2.rd2.distances.EuclideanDistance;

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
		else if (action.equals("verify"))
		{
			verify(args[1], args[2]);
		}
	}
	
	public static void verify(String predictedLabelPath, String actualLabelPath)
	{
		try (BufferedReader predictedReader = new BufferedReader(new InputStreamReader(new FileInputStream(predictedLabelPath)));
				BufferedReader actualReader = new BufferedReader(new InputStreamReader(new FileInputStream(actualLabelPath))))
		{
			int correct = 0, total = 0;
			for (String predicted = predictedReader.readLine(), actual = actualReader.readLine(); 
					predicted != null && actual != null; predicted = predictedReader.readLine(), actual = actualReader.readLine())
			{
				if (predicted.equals(actual))
					correct++;
				total++;
			}
			
			if (predictedReader.readLine() != null || actualReader.readLine() != null)
				throw new RuntimeException("File are of a different length!");
			
			double accuracy = (double)correct / (double)total;
			System.out.println("Accuracy: " + (accuracy * 100.0) + " %");
		} 
		catch (IOException e)
		{
			e.printStackTrace();
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
			// Determine number of attributes
			final int NUM_ATTRIBUTES = detectAttributeCount(trainPath);
			System.out.println("Detected " + NUM_ATTRIBUTES + " attributes");
			
			System.out.print("Loading " + trainPath + "...");
			trainingSet  = read(trainPath, scanner -> new Classification<AttributeVector, Integer>(
							AttributeVector.fromScanner(scanner, NUM_ATTRIBUTES), scanner.nextInt()));
			System.out.println(" done");
			
			/*
			if (trainingSet.size() > 7500)
			{
				System.out.print("Detected >7500 instances; taking random sample...");
				Collections.shuffle(trainingSet);
				trainingSet = trainingSet.subList(0, 100);
				System.out.println(" done");
			}
			*/
			
			System.out.print("Loading " + testPath + "...");
			testSet = read(testPath, scanner -> AttributeVector.fromScanner(scanner, NUM_ATTRIBUTES));
			System.out.println(" done");
		}
		catch (IOException ex)
		{
			throw new RuntimeException("File not found", ex);
		}
		
		long startTime = System.currentTimeMillis();
		
		DistanceMeasure<AttributeVector> distanceMeasure = new EuclideanDistance();
		
		// Run classifier
		System.out.print("Training classifier...");
		R2D2Classifier<AttributeVector, Integer> classifier = new R2D2Classifier<AttributeVector, Integer>(11, 51,
				distanceMeasure, distanceMeasure);
		classifier.train(trainingSet);
		System.out.println(" done");
		
		System.out.println("Discarded " + ((trainingSet.size() - classifier.getPrototypes().size()) / 
				(float)trainingSet.size() * 100) + "% as noise");

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

	private static int detectAttributeCount(String trainPath) throws FileNotFoundException, IOException
	{
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainPath)));
		final int NUM_ATTRIBUTES = reader.readLine().split(" ").length - 1;
		reader.close();
		
		return NUM_ATTRIBUTES;
	}
	
	/**
	 * Where we train weights, duh
	 * @param trainPath
	 */
	public static void gym(String trainPath)
	{
		try
		{
			// Load training set
			System.out.print("Loading " + trainPath + "...");
			final int dimensions = detectAttributeCount(trainPath);
			List<Classification<AttributeVector, Integer>> set  = read(trainPath, 
					scanner -> new Classification<AttributeVector, Integer>(
							AttributeVector.fromScanner(scanner, dimensions), scanner.nextInt()));
			System.out.println(" done");
			
			System.out.print("Filtering noise...");
			// train R2D2 for ENN
			R2D2Classifier<AttributeVector, Integer> c = new R2D2Classifier<AttributeVector, Integer>(11, 51, 
					new EuclideanDistance(), new EuclideanDistance());
			c.train(set);
			System.out.println(" done");

			// PUMP IT
			new PlantsVsWeightsGym(11).train(c.getPrototypes());
		}
		catch (IOException ex)
		{
			throw new RuntimeException("File not found: " + trainPath, ex);
		}
	}
	
	private static <T> List<T> read(String filePath, Function<Scanner, T> reader) throws FileNotFoundException
	{
		List<T> set = new ArrayList<T>();
		
		Scanner scanner = new Scanner(new FileInputStream(filePath));
		scanner.useLocale(Locale.US);
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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class Program
{
	public static class Vector2D
	{
		public float x;
		public float y;
		
		public Vector2D(float x, float y)
		{
			this.x = x;
			this.y = y;
		}
	}
	
	//private static final DistanceMeasure<String> distanceMeasure = (String s1, String s2) -> {
	//	return Math.abs(s2.length() - s1.length());
	//};
	
	private static final DistanceMeasure<Vector2D> distanceMeasure = (Vector2D a, Vector2D b) -> {
		float dx = a.x - b.x;
		float dy = a.y - b.y;
		return Math.sqrt(dx * dx + dy * dy);
	};
	
	public static void main(String[] args)
	{
		String trainSetPath = args[0];
		String testDataPath = args[1];
		String testLabelPath = args[2];
		
		try
		{
			// Load
			List<Classification<Vector2D, Integer>> trainingSet = read(trainSetPath, scanner -> 
				new Classification<Vector2D, Integer>(new Vector2D(scanner.nextFloat(), scanner.nextFloat()), scanner.nextInt()));
			List<Vector2D> testSet = read(testDataPath, scanner -> new Vector2D(scanner.nextFloat(), scanner.nextFloat()));
			
			// Classify
			Classifier<Vector2D, Integer> classifier = 
					new KNearestNeighborClassifier<Vector2D, Integer>(3, distanceMeasure);
			classifier.train(trainingSet);
			List<Classification<Vector2D, Integer>> predicted = classifier.classify(testSet);
			
			// Compare to test label data
			List<Integer> actual = read(testLabelPath, scanner -> scanner.nextInt());
			float classificationAccuracy = calculateAccuracy(predicted, actual);
			System.out.println(classificationAccuracy * 100.0 + "%");
		}
		catch (IOException ex)
		{
			throw new RuntimeException("Fail :(", ex);
		}
	}

	private static <D, C> float calculateAccuracy(List<Classification<D, C>> predicted, List<C> actual)
	{
		Iterator<Classification<D, C>> predictedIt = predicted.iterator();
		Iterator<C> actualIt = actual.iterator();
		
		
		int correct = 0;
		int incorrect = 0;
		while (predictedIt.hasNext() && actualIt.hasNext())
		{
			if (predictedIt.next().getClassLabel().equals(actualIt.next()))
				correct++;
			else
				incorrect++;
		}
		float classificationAccuracy = correct / (float)(correct + incorrect);
		return classificationAccuracy;
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
}

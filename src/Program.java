import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
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
	
	private static final DistanceMeasure<Vector2D> distanceMeasure = (Vector2D a, Vector2D b) -> {
		float dx = a.x - b.x;
		float dy = a.y - b.y;
		return Math.sqrt(dx * dx + dy * dy);
	};
	
	public static void main(String[] args)
	{
		List<Classification<Vector2D, Integer>> trainingSet;
		List<Vector2D> testSet;

		// Load
		try
		{
			String trainSetPath = args[0];
			String testDataPath = args[1];

			trainingSet = read(trainSetPath, scanner -> 
				new Classification<Vector2D, Integer>(
					new Vector2D(
						scanner.nextFloat(), 
						scanner.nextFloat()), 
					scanner.nextInt()));
			testSet = read(testDataPath, scanner -> 
				new Vector2D(
					scanner.nextFloat(), 
					scanner.nextFloat()));
		}
		catch (IOException ex)
		{
			throw new RuntimeException("SYSTEM FAILURE SYSTEM FAILURE SYSTEM FAILURE", ex);
		}
		
		// Classify
		Classifier<Vector2D, Integer> classifier = 
			new KNearestNeighborClassifier<Vector2D, Integer>(3, distanceMeasure);
		classifier.train(trainingSet);
		List<Classification<Vector2D, Integer>> predicted = classifier.classify(testSet);
		
		// Print result
		for (Classification<Vector2D, Integer> classification : predicted)
		{
			System.out.println(classification.getClassLabel());
		}
	}

	/**
	 * Compare predicted classifications with actual classifications and return
	 * the accuracy
	 */
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
	
	/**
	 * Read a number of T instances from a file using a Function<Scanner, T> to
	 * read individual instances
	 */
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

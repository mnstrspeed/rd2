import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class AttributeVector
	{
		private List<Double> elements;
		
		public AttributeVector(List<Double> elements)
		{
			this.elements = elements;
		}
		
		public Double get(int index)
		{
			return this.elements.get(index);
		}
		
		public int getDimension()
		{
			return this.elements.size();
		}
		
		@Override
		public String toString()
		{
			StringBuilder builder = new StringBuilder();
			for (int i = 0; i < this.getDimension(); i++)
			{
				builder.append(this.get(i));
				
				if (i + 1 < this.getDimension())
					builder.append(" ");
			}
			return builder.toString();
		}
		
		public static AttributeVector fromScanner(Scanner scanner, int dimension)
		{
			ArrayList<Double> buffer = new ArrayList<Double>();
			for (int i = 0; i < dimension; i++)
			{
				buffer.add(scanner.nextDouble());
			}
			return new AttributeVector(buffer);
		}
	}
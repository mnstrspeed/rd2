package r2d2.rd2.util;

public class StringHelper
{
	public static String join(double[] elements)
	{
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < elements.length; i++)
		{
			builder.append(elements[i]);
			if (i + 1 < elements.length)
				builder.append(" ");
		}
		return builder.toString();
	}
}

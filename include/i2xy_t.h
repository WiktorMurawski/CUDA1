// Struktura pozwalająca na szybki dostęp do współrzędnych (x, y) i-tego elementu w tablicy bez wykonywania obliczeń
struct i2xy_t
{
    int* x;
    int* y;

    i2xy_t(int width, int height)
    {
        int n = width * height;
        x = new int[n];
        y = new int[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = i % width;
            y[i] = i / width;
        }
    }

    ~i2xy_t()
    {
        delete[] x;
        delete[] y;
    }
};
// Struktura pozwalająca na szybki dostęp do współrzędnych (x, y) i-tego elementu w tablicy bez wykonywania obliczeń
struct i2xy_t
{
    int* x;
    int* y;

    i2xy_t(int width, int height)
    {
        int n = width * height;
        this->x = new int[n];
        this->y = new int[n];
        for (int i = 0; i < n; i++)
        {
            this->x[i] = i % width;
            this->y[i] = i / width;
        }
    }
    
    ~i2xy_t()
    {
        this->cleanup();
    }

    void cleanup(){
        delete[] this->x;
        delete[] this->y;
        this->x = nullptr;
        this->y = nullptr;
    }
};

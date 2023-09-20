// CircularBuffer.h

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

class CircularBuffer {
public:
    CircularBuffer(int size) {
        buffer = new char[size];
        bufferSize = size;
        head = 0;
        tail = 0;
    }
    void clearBuffer() {
        head = tail;
    }
    bool isFull() {
        //return ((head + 1) % bufferSize) == tail;
        return abs(head - tail) == bufferSize - 1;
    }
    bool isEmpty() {
        return head == tail;
    }
    void add(char data) {
        buffer[head] = data;
        head = (head + 1) % bufferSize;
    }
    char remove() {
        char data = buffer[tail];
        tail = (tail + 1) % bufferSize;
        return data;
    }
    void printBuffer() {
        int i = tail;
        while (i != head) {
            Serial.print(buffer[i]);
            i = (i + 1) % bufferSize;
        }
    }
    int bufferSize;
    int head;
    int tail;
private:
    char* buffer;
};

#endif

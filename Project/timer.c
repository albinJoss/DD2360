#include "timer.h"


void tstart(Timer_t *timer)
{
#ifdef _WIN32
    timespec_get(&timer->start, TIME_UTC);
#else
    gettimeofday(&timer->start, null);
#endif
}

void tend(Timer_t *timer)
{
#ifdef _WIN32
    timespec_get(&timer->end, TIME_UTC);
#else
gettimeofday(&timer->end, null);
#endif
}

double telapsed(Timer_t *timer)
{
#ifdef _WIN32
    return (double)(timer->end.tv_sec - timer->start.tv_sec) * 1000.0L + (double)(timer->end.tv_nsec - timer->start.tv_nsec) / 1000000.0L;;
#else
    return ((double) (timer->end.tv_sec - timer->start.tv_sec) + (double) (timer->end.tv_usec - timer->start.tv_usec) * 1.e-6);
#endif
}
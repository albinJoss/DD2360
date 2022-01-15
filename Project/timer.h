#ifndef TIMER_H_
#define TIMER_H_

#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

typedef struct Timer_t
{
    #ifdef _WIN32
    struct timespec start;
    struct timespec end;
    #else
    struct timeval strart;
    struct timeval end;
    #endif
} Timer_t;

#ifdef __cplusplus 
extern "C" void tstart(Timer_t *timer);
extern "C" void tend(Timer_t *timer);
extern "C" double telapsed(Timer_t *timer);
#endif

#endif

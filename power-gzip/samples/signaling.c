/* Simulates chained signal handlers */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* https://stackoverflow.com/questions/36044772/proper-way-to-chain-signal-handlers-in-linux */
/* https://cboard.cprogramming.com/linux-programming/117417-sigaction-how-find-invoke-previous-signal-handler-sa_handler.html */
/* https://stackoverflow.com/questions/10701713/signal-overwriting-other-signal-handlers */

/*
int sigaction(int signum, const struct sigaction *act,
struct sigaction *oldact);
The sigaction structure is defined as something like:

struct sigaction {
void     (*sa_handler)(int);
void     (*sa_sigaction)(int, siginfo_t *, void *);
sigset_t   sa_mask;
int        sa_flags;
void     (*sa_restorer)(void);
};
On some architectures a union is involved: do not assign to both sa_handler and sa_sigaction.
*/

static void install_new_handler(void *handler, struct sigaction *old, const char *comment)
{
	int rc;
	struct sigaction new_sigact;
	struct sigaction old_sigact;

	/* ./powerpc64le-linux-gnu/bits/sigaction.h */
	/* void sigsegv_handler(int sig, siginfo_t *info, void *ctx); */
	memset( &new_sigact, 0, sizeof(new_sigact) );
	memset( &old_sigact, 0, sizeof(old_sigact) );

	new_sigact.sa_sigaction = handler;
	new_sigact.sa_flags = SA_SIGINFO;

	if ( sigemptyset(&new_sigact.sa_mask) );   /* all signals unblocked */
		fprintf(stderr, "sigemptyset: %s\n", strerror(errno));

	if ( sigaction(SIGSEGV, &new_sigact, &old_sigact) )
		fprintf(stderr, "sigaction: %s\n", strerror(errno));

	/* copy the old handler to a safe place */
	if ( old != NULL )
		memcpy(old, &old_sigact, sizeof(old_sigact));

	fprintf(stderr, "installed handler %p for %s and retrieved old handler %p\n", handler, comment, old_sigact.sa_sigaction);
}

/* remove our handler from the chain */
static void restore_old_handler(struct sigaction *old)
{
	struct sigaction new_sigact;
	struct sigaction old_sigact;

	if ( sigaction(SIGSEGV, old, NULL) )
		fprintf(stderr, "sigaction: %s\n", strerror(errno));

	fprintf(stderr, "restored old handler %p\n", old->sa_sigaction);
}


/* Unfortunately I don't know how a private pointer may be passed to
   the handler to tell the old handler address, so using a global
   variable as suggested here
   https://cboard.cprogramming.com/linux-programming/117417-sigaction-how-find-invoke-previous-signal-handler-sa_handler.html
 */

struct sigaction make_believe_jvm_saved_this_old_handler;
void make_believe_jvm_handler(int sig, siginfo_t *info, void *ctx)
{
	fprintf(stderr, "%s(pid=%d): signal %d si_code %d, si_addr %p, info %p, ctx %p\n",
		__FUNCTION__, getpid(), sig, info->si_code, info->si_addr, (void *) info, (void *)ctx);
	fflush(stderr);

	/* nx_fault_storage_address = info->si_addr; */

	if ( make_believe_jvm_saved_this_old_handler.sa_sigaction != NULL ) {
		fprintf(stderr, "%s: calling the downstream handler %p\n",
			__FUNCTION__, make_believe_jvm_saved_this_old_handler.sa_sigaction);
		(make_believe_jvm_saved_this_old_handler.sa_sigaction)(sig, info, ctx);
	}

	fprintf(stderr, "\n");
}

struct sigaction our_library_saved_this_old_handler;
void our_library_sigsegv_handler(int sig, siginfo_t *info, void *ctx)
{
	fprintf(stderr, "%s(pid=%d): signal %d si_code %d, si_addr %p, info %p, ctx %p\n",
		__FUNCTION__, getpid(), sig, info->si_code, info->si_addr, (void *) info, (void *)ctx);
	fflush(stderr);
	/* nx_fault_storage_address = info->si_addr; */

	/* use random number generator to fake signal received from nx-gzip driver or something else */
	if ( sig == SIGSEGV && ((random() % 2) == 0) ) {
		/* pretend this handler understood the signal and will process it */
		fprintf(stderr, "%s: ++++ SIGNAL FOR US. We touch info->si_addr %p and return. Not calling downstream handler\n\n", __FUNCTION__, info->si_addr);
		return;
	}
	else {
		/* pretend this handler decided signal doesn't belong here and it will call the downstream handler which would be the JVM */
		fprintf(stderr, "%s: ---- SIGNAL *NOT* FOR US. Calling any found downstream handler %p\n", __FUNCTION__, our_library_saved_this_old_handler.sa_sigaction);
	}

	/* call the old action of the downstream handler */
	if ( our_library_saved_this_old_handler.sa_sigaction != NULL )
		(our_library_saved_this_old_handler.sa_sigaction)(sig, info, ctx);

	fprintf(stderr, "\n");
}

/*
   basic chaining
*/
int main1()
{
#if 1
	/* suppose JVM was running first */
	install_new_handler( make_believe_jvm_handler, &make_believe_jvm_saved_this_old_handler, "JVM");
	/* our_library comes later and inserts itself at the head of chain */
	install_new_handler( our_library_sigsegv_handler, &our_library_saved_this_old_handler, "OUR_LIBRARY" );
#else
	/* suppose our_library comes first  */
	install_new_handler( our_library_sigsegv_handler, &our_library_saved_this_old_handler, "OUR_LIBRARY" );
	/* then JVM installs itself */
	install_new_handler( make_believe_jvm_handler, &make_believe_jvm_saved_this_old_handler, "JVM" );
#endif

	fprintf( stderr, "our process id is %d : from another bash shell do several times: kill -SIGSEGV %d\n", getpid(), getpid());
	/* now we wait for signals */
	while(1)
		pause();
}


/*
   restoring old handler
*/
int main2()
{
	/* suppose JVM was running first */
	install_new_handler( make_believe_jvm_handler, &make_believe_jvm_saved_this_old_handler, "JVM" );
	/* our_library comes later and inserts itself at the head of chain */
	install_new_handler( our_library_sigsegv_handler, &our_library_saved_this_old_handler, "OUR_LIBRARY" );
	fprintf(stderr, "our process id is %d : from another bash shell do several times: kill -SIGSEGV %d, after 5 kills old handler will be restored\n", getpid(), getpid());

	/* now we wait for signals */
	int i=0;
	for (i = 0; i < 5; i++)
		pause();

	restore_old_handler( &our_library_saved_this_old_handler );

	while(1)
		pause();
}


/*
   fork experiment
*/
int main3()
{
#if 0
	/* suppose JVM was running first */
	install_new_handler( make_believe_jvm_handler, &make_believe_jvm_saved_this_old_handler, "JVM");
	/* our_library comes later and inserts itself at the head of chain */
	install_new_handler( our_library_sigsegv_handler, &our_library_saved_this_old_handler, "OUR_LIBRARY" );
#else
	/* suppose our_library comes first  */
	install_new_handler( our_library_sigsegv_handler, &our_library_saved_this_old_handler, "OUR_LIBRARY" );
	/* then JVM installs itself */
	install_new_handler( make_believe_jvm_handler, &make_believe_jvm_saved_this_old_handler, "JVM" );
#endif

	if (fork() == 0)
		fprintf(stderr, "child (pid=%d)\n", getpid());
	else
		fprintf(stderr, "parent (pid=%d)\n", getpid());

	fprintf( stderr, "our process id is %d : from another bash shell do several times: kill -SIGSEGV %d\n", getpid(), getpid());
	/* now we wait for signals */
	while(1)
		pause();
}


int main()
{
	return main3();
}

## Command Recording

Merian provides several abstractions and helpers for command recording.

- `Queue`: Holds a queue together with its mutex. Provides convenience methods to submit using the mutex.
- `CommandPool`: Wraps a command pool that is automatically destroyed when the object is destroyed.

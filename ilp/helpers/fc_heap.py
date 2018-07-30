import heapq
import warnings


class FixedCapacityHeap:
    """Implementation of a min-heap with fixed capacity.
    The heap contains tuples of the form (edge_weight, node_id), 
    which means the min. edge weight is extracted first
    """

    def __init__(self, lst=None, capacity=10):
        self.capacity = capacity
        if lst is None:
            self.data = []
        elif type(lst) is list:
            self.data = lst
        else:
            self.data = lst.tolist()

        if lst is not None:
            heapq.heapify(self.data)
            if len(self.data) > capacity:
                msg = 'Input data structure is larger than the queue\'s ' \
                      'capacity ({}), truncating to smallest ' \
                      'elements.'.format(capacity)

                warnings.warn(msg, UserWarning)
                self.data = self.data[:self.capacity]

    def push(self, item):
        """Insert an element in the heap if its key is smaller than the current 
        max-key elements and remove the current max-key element if the new 
        heap size exceeds the heap capacity

        Args:
            item (tuple): (edge_weight, node_ind)

        Returns:
            tuple :  (bool, item)
                bool: whether the item was actually inserted in the queue
                item: another item that was removed from the queue or None if none was removed

        """
        inserted = False
        removed = None
        if len(self.data) < self.capacity:
            heapq.heappush(self.data, item)
            inserted = True
        else:
            if item > self.get_min():
                removed = heapq.heappushpop(self.data, item)
                inserted = True

        return inserted, removed

    def get_min(self):
        """Return the min-key element without removing it from the heap"""
        return self.data[0]

    def __len__(self):
        return len(self.data)

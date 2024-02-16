Logger = Callable[[Any], None]

CollatableProps = list[Any] | Tensor

InputType = TypeVar('InputType')

class AbstractBatchInput:
    _list_keys: list[str] = []
    _tensor_keys: dict[str, tuple[int, ...]] = {}
    
    def __init__(
        self,
        **kwargs : CollatableProps
    ) -> None:
        for key in self.__class__._list_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])
        for key in self.__class__._tensor_keys:
            if key not in kwargs:
                raise ValueError(f"Key {key} is not present in {kwargs}")
            setattr(self, key, kwargs[key])

    @classmethod
    def collate_fn(cls: Type[InputType], batch: Sequence["AbstractColorPrompt"]) -> InputType:
        opts : dict[str, CollatableProps] = {}
        for key in cls._list_keys:

            opts[key] : list[Any] = []

            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                for prop in getattr(item, key):
                    opts[key].append(prop)
        for key in cls._tensor_keys:
            
            lst : list[Tensor] = []
            for item in batch:
                if not hasattr(item, key):
                    raise ValueError(f"Key {key} is not present in {item}")
                tensor = getattr(item, key)
                if not isinstance(tensor, Tensor):
                    raise ValueError(f"Key {key}, {tensor} should be a tensor")
                lst.append(tensor)
            
            opts[key] = cat(lst)

        return cls(**opts)
    
    @classmethod
    def empty(cls: Type[InputType]) -> InputType:
        opts : dict[str, CollatableProps] = {}
        
        for key in cls._list_keys:
            opts[key] = []
        for key in cls._tensor_keys:
            size = cls._tensor_keys[key]
            tensor = empty((0,)+ size)
            opts[key] = tensor
            
        return cls(**opts)

    def get_indices(self: InputType, indices: list[int]) -> InputType:
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__class__._list_keys:
            opts[key] = [getattr(self, key)[i] for i in indices]
        for key in self._tensor_keys:
            opts[key] = getattr(self, key)[indices]
            
        return self.__class__(**opts)
    
    def get_prompt(self: InputType, prompt: str) -> InputType:
        prompts = cast(list[str], getattr(self, "source_prompts"))
        indices = [i for i, p in enumerate(prompts) if p == prompt]
        return self.get_indices(indices)
    
class AbstractBatchOutput(Generic[InputType], AbstractBatchInput):
    __prompt_type: Type[InputType]
    
    def to_input(self) -> InputType:
        opts : dict[str, CollatableProps] = {}
        
        for key in self.__prompt_type._list_keys:
            opts[key] = getattr(self, key)
        for key in self.__prompt_type._tensor_keys:
            opts[key] = getattr(self, key)
        
        return self.__prompt_type()

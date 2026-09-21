class TinygradExportArchive:
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the tinygrad backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the tinygrad backend."
        )


# `ExportArchive`: the older plugin-PoC name.
ExportArchive = TinygradExportArchive

# keras' pluggable_backend branch reads `SavedModelExportArchive` from
# `keras_<name>.src.export` when the module exists and uses it AS the
# archive class (keras 3.15.x instead subclasses the backend class above).
# Its base only exists on the branch; both backend hooks raise a loud
# NotImplementedError there, like TinygradExportArchive does.
try:
    from keras.src.export.saved_model_export_archive import (
        BaseSavedModelExportArchive,
    )
except ImportError:  # keras 3.15.x: no base class of that name
    BaseSavedModelExportArchive = None

if BaseSavedModelExportArchive is not None:

    class SavedModelExportArchive(BaseSavedModelExportArchive):
        pass
